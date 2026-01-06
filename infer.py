import os, json, cv2
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import get_config_file

# ============================================
# Fallback mapping (si jamais COCO_JSON absent)
# ============================================
FALLBACK_CLASS_NAMES = {
    0: "headlamp",
    1: "rear_bumper",
    2: "door",
    3: "hood",
    4: "front_bumper",
}

# =========================
# PARAMS (env vars)
# =========================
IMG = os.environ.get("IMG", "/workspace/test.jpg")
WEIGHTS = os.environ.get("WEIGHTS", "/workspace/part_segmentation_model.pth")
COCO_JSON = os.environ.get("COCO_JSON", "/workspace/COCO_mul_train_annos.json")

SCORE_THRESH = float(os.environ.get("SCORE", "0.3"))
OUT_IMG = os.environ.get("OUT_IMG", "/workspace/result.png")
OUT_JSON = os.environ.get("OUT_JSON", "/workspace/preds.json")


def load_coco_contig_mapping(coco_json_path: str):
    """
    Retourne un mapping contiguous_id (0..N-1) -> name
    basé sur les 'categories' du COCO JSON triées par id.
    Si le fichier n'existe pas, fallback sur FALLBACK_CLASS_NAMES.
    """
    if not coco_json_path or not os.path.exists(coco_json_path):
        contig_to_name = dict(sorted(FALLBACK_CLASS_NAMES.items(), key=lambda x: x[0]))
        print(f"⚠️ COCO_JSON introuvable -> fallback mapping: {contig_to_name}")
        return contig_to_name

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    cats = coco.get("categories", [])
    if not cats:
        contig_to_name = dict(sorted(FALLBACK_CLASS_NAMES.items(), key=lambda x: x[0]))
        print(f"⚠️ 'categories' vide -> fallback mapping: {contig_to_name}")
        return contig_to_name

    cats_sorted = sorted(cats, key=lambda x: x["id"])
    # contiguous id = index dans cats_sorted
    contig_to_name = {i: c["name"] for i, c in enumerate(cats_sorted)}
    return contig_to_name


# =========================
# 1) READ TRUE MAPPING
# =========================
contig_to_name = load_coco_contig_mapping(COCO_JSON)
N_COCO = len(contig_to_name)
print("✅ COCO classes:", N_COCO, contig_to_name)

# =========================
# 2) SETUP DETECTRON2 CFG
# =========================
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
cfg.MODEL.WEIGHTS = WEIGHTS

# Déduire K depuis le checkpoint pour éviter les "Skip loading parameter ..."
ckpt = torch.load(WEIGHTS, map_location="cpu")
sd = ckpt.get("model", ckpt)

cls_w = sd.get("roi_heads.box_predictor.cls_score.weight", None)
if cls_w is not None:
    K = int(cls_w.shape[0] - 1)  # (K+1, 1024)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = K
    print(f"✅ Model expects NUM_CLASSES={K} (from checkpoint)")
else:
    K = N_COCO
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = K
    print(f"⚠️ Could not read checkpoint class count -> NUM_CLASSES={K}")

# =========================
# 3) METADATA (labels)
# IMPORTANT: metadata doit être cohérente avec ce qu'on va afficher.
# Ici, on affiche seulement les classes COCO (on filtrera le phantom),
# donc thing_classes = N_COCO labels.
# =========================
meta_name = "car_parts_runtime"
metadata = MetadataCatalog.get(meta_name)
metadata.set(thing_classes=[contig_to_name[i] for i in range(N_COCO)])

# =========================
# 4) INFER
# =========================
predictor = DefaultPredictor(cfg)

im = cv2.imread(IMG)
if im is None:
    raise FileNotFoundError(f"Image not found: {IMG}")

outputs = predictor(im)
inst = outputs["instances"].to("cpu")

# =========================
# 5) FILTER phantom classes
# - N_COCO = classes réelles (ex: 5)
# - K peut être 6 (phantom)
# On ne garde que cls in [0, N_COCO-1]
# =========================
keep_idx = []
preds = []

for i in range(len(inst)):
    cls = int(inst.pred_classes[i])
    score = float(inst.scores[i])

    if cls < 0 or cls >= N_COCO:
        continue  # ignore phantom

    bbox = inst.pred_boxes[i].tensor.numpy().tolist()[0]  # [x1,y1,x2,y2]

    # mask_area utile si tu veux estimer "ampleur"
    mask_area = None
    if hasattr(inst, "pred_masks") and inst.pred_masks is not None:
        mask = inst.pred_masks[i].numpy().astype("uint8")
        mask_area = int(mask.sum())

    keep_idx.append(i)
    preds.append({
        "cls_id": cls,
        "label": contig_to_name.get(cls, "unknown"),
        "score": score,
        "bbox_xyxy": bbox,
        "mask_area": mask_area,
    })

inst_f = inst[keep_idx] if keep_idx else inst[:0]

# =========================
# 6) VISUALIZE
# =========================
vis = Visualizer(
    im[:, :, ::-1],
    metadata=metadata,
    scale=1.0,
    instance_mode=ColorMode.IMAGE
)
out = vis.draw_instance_predictions(inst_f)
res = out.get_image()[:, :, ::-1]
cv2.imwrite(OUT_IMG, res)

# =========================
# 7) EXPORT JSON
# =========================
payload = {
    "image": IMG,
    "weights": WEIGHTS,
    "score_thresh": SCORE_THRESH,
    "num_preds": len(preds),
    "preds": preds,
    "coco_mapping_contig": contig_to_name,  # 0..N-1 -> name
    "note": "Predictions filtered to COCO classes to ignore phantom class if any.",
    "debug": {
        "N_COCO": N_COCO,
        "K_model_num_classes": K,
    }
}

with open(OUT_JSON, "w") as f:
    json.dump(payload, f, indent=2)

print(f"✅ Done -> {OUT_IMG}")
print(f"✅ JSON -> {OUT_JSON}")
