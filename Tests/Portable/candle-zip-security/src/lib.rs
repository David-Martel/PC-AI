//! Verifies ZIP header integrity through each supported Candle consumer.

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    /// Changes only the central directory filename of a valid tensor archive.
    fn change_directory_name(path: &Path) {
        let mut data = fs::read(path).expect("read owned archive");
        let position = data
            .windows(4)
            .position(|signature| signature == b"PK\x01\x02")
            .expect("valid ZIP central directory");
        let length = usize::from(u16::from_le_bytes([data[position + 28], data[position + 29]]));
        assert_eq!(&data[position + 46..position + 46 + length], b"values.npy");
        data[position + 46..position + 46 + length].copy_from_slice(b"hidden.npy");
        fs::write(path, data).expect("write owned malformed archive");
    }

    /// Changes one filename in a selected header without changing entry data.
    fn change_fixture_header_name(
        path: &Path,
        signature: [u8; 4],
        name_offset: usize,
        length_offset: usize,
        original: &[u8],
        replacement: &[u8],
    ) {
        assert_eq!(original.len(), replacement.len());
        let mut data = fs::read(path).expect("read owned PyTorch fixture");
        let position = data
            .windows(4)
            .enumerate()
            .find_map(|(offset, bytes)| {
                if bytes != signature {
                    return None;
                }
                let length = usize::from(u16::from_le_bytes([
                    data[offset + length_offset],
                    data[offset + length_offset + 1],
                ]));
                (data.get(offset + name_offset..offset + name_offset + length) == Some(original)).then_some(offset)
            })
            .expect("the original fixture header exists");
        data[position + name_offset..position + name_offset + original.len()].copy_from_slice(replacement);
        fs::write(path, data).expect("write owned malformed PyTorch fixture");
    }

    macro_rules! consumer_contract {
        ($module:ident, $candle:ident) => {
            mod $module {
                use super::*;
                use $candle::{npy::NpzTensors, Device, Tensor};

                fn valid_archive(path: &Path) {
                    let tensor = Tensor::new(&[1.0_f32, 2.0_f32], &Device::Cpu).expect("owned CPU tensor");
                    Tensor::write_npz(&[("values", &tensor)], path).expect("write valid owned tensor archive");
                }

                // Protects: unchanged eager tensor names, shape and numeric data on all three Candle versions.
                // Detects: a ZIP upgrade that rejects valid NPZ or changes loaded tensor values.
                // Needs: owned temporary files and CPU-only Candle/ZIP; no model, GPU or network.
                // Breadcrumb: vendored candle-core src/npy.rs Tensor::read_npz/write_npz.
                #[test]
                fn eager_valid_tensor_round_trip() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    let tensors = Tensor::read_npz(&path).expect("read valid archive");
                    assert_eq!(tensors.len(), 1);
                    assert_eq!(tensors[0].0, "values");
                    assert_eq!(tensors[0].1.to_vec1::<f32>().expect("vector"), vec![1.0, 2.0]);
                }

                // Protects: lazy tensor indexing and data retrieval preserve the existing public API.
                // Detects: changed ZIP names or index semantics after dependency adaptation.
                // Needs: owned NPZ file and CPU only, across 0.9.2/0.10.2/0.11.0.
                // Breadcrumb: vendored candle-core src/npy.rs NpzTensors::new/get.
                #[test]
                fn lazy_valid_tensor_round_trip() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    let archive = NpzTensors::new(&path).expect("valid lazy archive");
                    let tensor = archive.get("values").expect("read tensor").expect("named tensor");
                    assert_eq!(tensor.to_vec1::<f32>().expect("vector"), vec![1.0, 2.0]);
                }

                // Protects: selecting a named tensor preserves its data and shape.
                // Detects: ZIP by-name API adaptation changing valid archive lookup.
                // Needs: owned valid NPZ and CPU only; no model or native DLL.
                // Breadcrumb: vendored candle-core src/npy.rs Tensor::read_npz_by_name.
                #[test]
                fn selected_valid_tensor_round_trip() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    let tensors = Tensor::read_npz_by_name(&path, &["values"]).expect("read selected tensor");
                    assert_eq!(tensors.len(), 1);
                    assert_eq!(tensors[0].dims(), &[2]);
                    assert_eq!(tensors[0].to_vec1::<f32>().expect("vector"), vec![1.0, 2.0]);
                }

                // Protects: explicit named lookups refuse a conflicting directory alias.
                // Detects: strict validation present only in by-index entrypoints.
                // Needs: owned NPZ with equal-length central filename mutation.
                // Breadcrumb: vendored candle-core src/npy.rs Tensor::read_npz_by_name.
                #[test]
                fn selected_rejects_dual_header_filename_mismatch() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    change_directory_name(&path);
                    assert!(Tensor::read_npz_by_name(&path, &["hidden"]).is_err());
                }

                // Protects: lazy readers validate headers again after reopening a file.
                // Detects: validating only construction while later tensor/shape reads trust a replaced archive.
                // Needs: owned CPU NPZ, valid lazy index and a later central-name mutation.
                // Breadcrumb: vendored candle-core src/npy.rs NpzTensors::get/get_shape_and_dtype.
                #[test]
                fn lazy_reopened_readers_reject_later_header_mismatch() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    let archive = NpzTensors::new(&path).expect("valid initial index");
                    let (shape, dtype) = archive
                        .get_shape_and_dtype("values")
                        .expect("valid shape and dtype");
                    assert_eq!(shape.dims(), &[2]);
                    assert_eq!(dtype, $candle::DType::F32);
                    change_directory_name(&path);
                    assert!(archive.get("values").is_err());
                    assert!(archive.get_shape_and_dtype("values").is_err());
                }

                // Protects: harmless non-PyTorch ZIP entries remain a successful empty tensor listing.
                // Detects: changed pickle enumeration rejecting an otherwise valid archive.
                // Needs: valid owned NPZ and the real CPU-only PyTorch metadata reader.
                // Breadcrumb: vendored candle-core src/pickle.rs read_pth_tensor_info.
                #[test]
                fn pickle_valid_non_tensor_entries_are_ignored() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    let tensors =
                        $candle::pickle::read_pth_tensor_info(&path, false, None).expect("valid archive metadata");
                    assert!(tensors.is_empty());
                }

                // Protects: a published real PyTorch archive still loads its tensor data.
                // Detects: metadata enumeration or by-name ZIP adaptation breaking valid pickle/storage reads.
                // Needs: unchanged Candle 0.11.0 test.pt fixture, owned copy and CPU only.
                // Breadcrumb: vendored candle-core src/pickle.rs read_pth_tensor_info/PthTensors::get.
                #[test]
                fn pickle_valid_tensor_round_trip() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.pt");
                    fs::write(&path, include_bytes!("../fixtures/test.pt")).expect("owned fixture");
                    let info =
                        $candle::pickle::read_pth_tensor_info(&path, false, None).expect("valid PyTorch metadata");
                    assert_eq!(info.len(), 1);
                    assert_eq!(info[0].name, "test");
                    let archive = $candle::pickle::PthTensors::new(&path, None).expect("valid index");
                    let tensor = archive.get("test").expect("tensor read").expect("named tensor");
                    assert_eq!(tensor.dims(), &[2, 4]);
                    assert_eq!(
                        tensor.to_vec2::<i64>().expect("tensor data"),
                        vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
                    );
                }

                // Protects: conflicting metadata names fail instead of becoming successful empty metadata.
                // Detects: accepting a changed central alias while ignoring a real data.pkl entry.
                // Needs: owned published PyTorch fixture with central name alone changed.
                // Breadcrumb: vendored candle-core src/pickle.rs strict enumeration of every ZIP entry.
                #[test]
                fn pickle_real_metadata_rejects_header_mismatch() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.pt");
                    fs::write(&path, include_bytes!("../fixtures/test.pt")).expect("owned fixture");
                    change_fixture_header_name(&path, *b"PK\x01\x02", 46, 28, b"test/data.pkl", b"test/evil.pkl");
                    assert!($candle::pickle::read_pth_tensor_info(&path, false, None).is_err());
                }

                // Protects: lazy PyTorch storage reads validate headers when reopening the file.
                // Detects: a local storage-name change hidden behind the original central alias.
                // Needs: valid owned PyTorch index and subsequent local-header-only mutation.
                // Breadcrumb: vendored candle-core src/pickle.rs PthTensors::get by-name read.
                #[test]
                fn pickle_reopened_storage_rejects_later_header_mismatch() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.pt");
                    fs::write(&path, include_bytes!("../fixtures/test.pt")).expect("owned fixture");
                    let archive = $candle::pickle::PthTensors::new(&path, None).expect("valid index");
                    change_fixture_header_name(&path, *b"PK\x03\x04", 30, 26, b"test/data/0", b"test/data/1");
                    assert!(archive.get("test").is_err());
                }

                // Protects: eager loaders refuse conflicting local and central filenames.
                // Detects: an archive admitted using only its benign central directory alias.
                // Needs: a valid owned tensor ZIP with only the central filename changed.
                // Breadcrumb: vendored candle-core src/npy.rs fallible ZipFile::name propagation.
                #[test]
                fn eager_rejects_dual_header_filename_mismatch() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    change_directory_name(&path);
                    assert!(Tensor::read_npz(&path).is_err(), "ambiguous archive was admitted");
                }

                // Protects: lazy indexes do not retain a conflicting central directory alias.
                // Detects: swallowing filename validation errors during index construction.
                // Needs: owned valid tensor ZIP, modified only at the central filename.
                // Breadcrumb: vendored candle-core src/npy.rs NpzTensors::new.
                #[test]
                fn lazy_rejects_dual_header_filename_mismatch() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    change_directory_name(&path);
                    assert!(NpzTensors::new(&path).is_err(), "ambiguous index was admitted");
                }

                // Protects: PyTorch metadata enumeration propagates every ZIP filename validation failure.
                // Detects: dropping an invalid entry or returning empty successful metadata.
                // Needs: an owned valid archive with one conflicting central header; no pickle code executes.
                // Breadcrumb: vendored candle-core src/pickle.rs fallible file_names collection.
                #[test]
                fn pickle_rejects_dual_header_filename_mismatch() {
                    let scratch = tempfile::tempdir().expect("owned directory");
                    let path = scratch.path().join("tensor.npz");
                    valid_archive(&path);
                    change_directory_name(&path);
                    assert!(
                        $candle::pickle::read_pth_tensor_info(&path, false, None).is_err(),
                        "ambiguous entry was discarded as successful empty metadata"
                    );
                }
            }
        };
    }

    // Explicit selector groups: all twelve consumer contracts apply to each locked version.
    consumer_contract!(candle_09, candle09);
    consumer_contract!(candle_10, candle10);
    consumer_contract!(candle_11, candle11);
}
