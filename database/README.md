# Database Directory

This directory contains the FAISS indexes and ID mappings for the HCMAI 2025 Image Search project.

## Required Files

For the application to work properly, you need the following files:

```
database/
├── clip_faiss.bin         # FAISS index for CLIP model
├── clip_id_map.json        # ID to image path mapping for CLIP
├── openclip_faiss.bin      # FAISS index for OpenCLIP model
└── openclip_id_map.json    # ID to image path mapping for OpenCLIP
```

## Generating Index Files

To generate these files, run the following command:

```bash
python build_index.py --data_dir data/keyframes --output_dir database
```

## File Formats

- `*_faiss.bin`: Binary FAISS index files
- `*_id_map.json`: JSON files mapping numeric IDs to image paths relative to the project root

## Notes

- These files are not included in version control due to their size
- The application will create these files automatically if they don't exist
- Make sure the image paths in the ID maps are relative to the project root
