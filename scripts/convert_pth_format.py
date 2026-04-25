import sys
import torch

def main():
    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python convert_weights.py <input_legacy.pth> <output_modern.pth>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # Load the legacy pth file
        state_dict = torch.load(input_file, map_location='cpu', weights_only=False)
        
        # Save it using the modern ZIP format
        torch.save(state_dict, output_file)
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()