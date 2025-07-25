import argparse
import os


if __name__ == "__main__":
    # Parent parser
    parent_parser = argparse.ArgumentParser(add_help=False)

    # Subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task", required=True)
    
    # Shot boundary detection
    parser_shot_boundary_detection = subparsers.add_parser("shot_boundary_detection", parents=[parent_parser])
    parser_shot_boundary_detection.add_argument("mode", choices=["all", "lesson"])
    parser_shot_boundary_detection.add_argument("input_video_dir", type=str)
    parser_shot_boundary_detection.add_argument("output_shot_dir", type=str)
    parser_shot_boundary_detection.add_argument("--lesson_name", type=str)
    
    
    # Keyframe extraction
    parser_keyframe_extraction = subparsers.add_parser("keyframe_extraction", parents=[parent_parser])
    parser_keyframe_extraction.add_argument("mode", choices=["all", "lesson"])
    parser_keyframe_extraction.add_argument("input_video_dir", type=str)
    parser_keyframe_extraction.add_argument("input_shot_dir", type=str)
    parser_keyframe_extraction.add_argument("output_keyframe_dir", type=str)
    parser_keyframe_extraction.add_argument("--lesson_name", type=str)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run task
    if args.task == "shot_boundary_detection":
        # Check error
        if not os.path.exists(args.input_video_dir):
            raise ValueError("Input video directory does not exist")
        
        if args.mode == "lesson":
            if not args.lesson_name:
                raise ValueError("Lesson name is required when mode is lesson")
            if not os.path.exists(os.path.join(args.input_shot_dir, args.lesson_name)):
                raise ValueError("Lesson directory does not exist")
            
        # Process
        from preprocess.shot_boundary_detection import detect_shot_boundary
        if args.mode == "all":
            detect_shot_boundary(args.input_video_dir, args.output_shot_dir, args.mode)
        elif args.mode == "lesson":
            detect_shot_boundary(args.input_video_dir, args.output_shot_dir, args.mode, args.lesson_name)
        
    elif args.task == "keyframe_extraction":
        # Check error
        if not os.path.exists(args.input_video_dir):
            raise ValueError("Input video directory does not exist")
        
        if not os.path.exists(args.input_shot_dir):
            raise ValueError("Input shot directory does not exist")
        
        if args.mode == "lesson":
            if not args.lesson_name:
                raise ValueError("Lesson name is required when mode is lesson")
            
        # Process
        from preprocess.keyframe_extraction import extract_keyframe
        if args.mode == "all":
            extract_keyframe(args.input_video_dir, args.input_shot_dir, args.output_keyframe_dir, args.mode)
        elif args.mode == "lesson":
            extract_keyframe(args.input_video_dir, args.input_shot_dir, args.output_keyframe_dir, args.mode, args.lesson_name)
    else:
        raise ValueError(f"Invalid task: {args.task}")