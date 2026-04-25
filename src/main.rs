use burn::Tensor;
use burn::backend::NdArray;
use burn_std::network::downloader::download_file_as_bytes;
use burn_store::pytorch::PytorchReader;
use std::path::Path;
use std::process::Command;
use std::{
    fs::{File, rename},
    io::Write,
    path::PathBuf,
};

const VGG16_URL: &str = "https://download.pytorch.org/models/vgg16-397923af.pth";

/// Calls the Python script using `uv run`, automatically handling the PyTorch dependency.
fn convert_legacy_pth_to_zip(
    script_path: &Path,
    input_path: &Path,
    output_path: &Path,
) -> Result<(), String> {
    // uv will automatically ensure an isolated environment with `torch` exists before running.
    let output = Command::new("uv")
        .arg("run")
        .arg("--with")
        .arg("torch")
        .arg("python")
        .arg(script_path)
        .arg(input_path)
        .arg(output_path)
        .output()
        .map_err(|e| format!("Failed to execute uv command: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python conversion failed:\n{}", stderr));
    }

    Ok(())
}

fn convert_pth_format(legacy_file: &PathBuf, modern_file: &PathBuf) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let script_file: PathBuf = [manifest_dir, "scripts", "convert_pth_format.py"]
        .iter()
        .collect();

    println!("Converting legacy weights using uv...");
    match convert_legacy_pth_to_zip(&script_file, legacy_file, modern_file) {
        Ok(_) => println!("Conversion successful! File saved to {:?}", modern_file),
        Err(e) => eprintln!("Failed to convert: {}", e),
    }
}

fn download_pretrained_weights(cache_path: &PathBuf) {
    if !cache_path.exists() {
        let bytes = download_file_as_bytes(
            VGG16_URL,
            "Downloading VGG pretrained weights from PyTorch...",
        );

        // Write to a temporary file. If writing was completed, then rename to the correct name.
        // If writing is not completed, the cache file with the correct name (i.e. `cache_path`) will
        // not exist so this code block will run again when this function gets called again.
        let temp_path = cache_path.with_extension("pth.temp");
        let mut file = File::create(&temp_path).expect("Failed to create a VGG model cache file");
        file.write_all(&bytes)
            .expect("Failed to write pretrained VGG weights to the cache file");
        rename(temp_path, cache_path)
            .expect("Failed to rename temporary file to the correct VGG19 cache file name");
    }
}

fn print_pth_metadata_and_weights(pth_file_path: &PathBuf) {
    let reader = PytorchReader::new(pth_file_path).unwrap();
    println!("\n\n{:#?}", reader.metadata());

    // Print the weights of the very first convolution layer
    let data = reader.get("features.0.weight").unwrap().to_data().unwrap();
    let device = Default::default();
    let weight_tensor = Tensor::<NdArray, 4>::from_data(data, &device);
    let pth_file_name = pth_file_path.file_name();
    println!("{:?}:", pth_file_name);
    println!("First conv layer: {}\n\n", weight_tensor);
}

pub fn main() {
    let project_root_str = env!("CARGO_MANIFEST_DIR");
    let project_root = PathBuf::from(project_root_str);
    let legacy_path = project_root.join("vgg16_legacy.pth");
    let modern_path = project_root.join("vgg16_modern.pth");

    download_pretrained_weights(&legacy_path);
    print_pth_metadata_and_weights(&legacy_path);

    convert_pth_format(&legacy_path, &modern_path);
    print_pth_metadata_and_weights(&modern_path);
}
