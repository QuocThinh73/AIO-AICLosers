import os
import json

def get_places365_classes():
    """Get all Places365 class names"""
    classes_file = os.path.join("database", "places365_classes.json")
    with open(classes_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    classes = get_places365_classes()
    for class_name in classes:
        print(class_name)

if __name__ == "__main__":
    main() 