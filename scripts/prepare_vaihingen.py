"""
Convert the raw ISPRS Vaihingen release into the layout used by wsss:

    <out>/images/area{id}.tif         IRRG orthophotos (copied)
    <out>/labels/area{id}.png         class indices (0-5)
    <out>/labels_eroded/area{id}.png  class indices, boundary pixels = 255

Usage:
    python scripts/prepare_vaihingen.py \
        --images  <raw>/top \
        --labels  <raw>/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE \
        --eroded  <raw>/ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE \
        --out data/vaihingen
"""
import argparse
import os
import re
import shutil

import numpy as np
from PIL import Image
from skimage import io

from wsss.constants import ERODED_LABEL_NAME_FORMAT, IGNORE_INDEX, IMAGE_NAME_FORMAT
from wsss.labels import convert_from_color

EXPECTED_AREAS = 33


def area_ids(folder):
    pattern = re.compile(r"^top_mosaic_09cm_area(\d+)\.tif$")
    return sorted((m.group(1) for m in map(pattern.match, os.listdir(folder)) if m), key=int)


def convert_label(source, destination):
    labels = convert_from_color(io.imread(source)[..., :3])
    Image.fromarray(labels).save(destination)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="folder of top_mosaic_09cm_area*.tif")
    parser.add_argument("--labels", required=True, help="folder of the complete RGB ground truth")
    parser.add_argument("--eroded", default=None, help="folder of the eroded RGB ground truth")
    parser.add_argument("--out", default="data/vaihingen")
    args = parser.parse_args()

    ids = area_ids(args.labels)
    if len(ids) != EXPECTED_AREAS:
        print("Warning: found {} labelled areas, expected {}".format(len(ids), EXPECTED_AREAS))
    for folder in ("images", "labels", "labels_eroded"):
        os.makedirs(os.path.join(args.out, folder), exist_ok=True)

    for area_id in ids:
        image_source = os.path.join(args.images, IMAGE_NAME_FORMAT.format(area_id))
        if not os.path.isfile(image_source):
            raise FileNotFoundError(image_source)
        shutil.copyfile(image_source,
                        os.path.join(args.out, "images", "area{}.tif".format(area_id)))
        labels = convert_label(os.path.join(args.labels, IMAGE_NAME_FORMAT.format(area_id)),
                               os.path.join(args.out, "labels", "area{}.png".format(area_id)))
        unknown = np.mean(labels == IGNORE_INDEX)
        if unknown > 0:
            print("Warning: area {} has {:.4%} pixels of unknown color".format(area_id, unknown))
        if args.eroded is not None:
            convert_label(os.path.join(args.eroded, ERODED_LABEL_NAME_FORMAT.format(area_id)),
                          os.path.join(args.out, "labels_eroded",
                                       "area{}.png".format(area_id)))
        print("area {} done".format(area_id))
    print("Prepared {} areas in {}".format(len(ids), args.out))


if __name__ == "__main__":
    main()
