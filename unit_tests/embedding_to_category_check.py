def check_metadata_categories(metadata):
    missing_classes = []
    for ann_id, ann in metadata.anns.items():
        if not ann["mod_class_from_embd"]:
            missing_classes.append(ann_id)
            print("Missing modified class name at annotation: {ann_id}")
    if len(missing_classes) > 0:
        raise ValueError(f"Test Failed: {len(missing_classes)} Annotations are Missing Classes")
    else:
        print(f"Test Passed: Each Annotation has a Class")