"""Hash custody for the media weight-safety test registers.

Two registers bind individual test selectors to LF-normalised SHA-256 hashes:
`media-worker-lock-review.json` (Python selectors) and
`media-weight-safety-review.json` (Rust selectors). A register nobody can fail
is paper, so these checks recompute every hash from the working tree and fail on
drift, on an unbound Python test, and on any entry that claims more review than
it has. They need no Cargo, model, GPU or network.

Hashing rules, matching the registers' `hash_normalization` field:
- files: UTF-8 bytes with CRLF converted to LF;
- Python selectors: `ast.get_source_segment` of the test function;
- Rust selectors: the contiguous comment and attribute lines above the function
  (including the test attribute) through the function's closing brace at its own
  indent, plus a final newline.
"""

# The expected selector lists hold long identifiers copied from the Rust source; wrapping
# them would make the lists harder to diff against the registers, so E501 is off here.
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
PORTABLE = ROOT / "Tests" / "Portable"
PYTHON_REGISTER = PORTABLE / "media-worker-lock-review.json"
RUST_REGISTER = PORTABLE / "media-weight-safety-review.json"

TEST_ATTR = re.compile(r"^\s*#\[(?:tokio::)?test(?:\(.*\))?\]\s*$")
FN = re.compile(r"^(\s*)(?:async\s+)?fn\s+([A-Za-z0-9_]+)\s*\(")
MOD = re.compile(r"^(\s*)(?:pub\s+)?mod\s+([A-Za-z0-9_]+)\s*\{")


def read_lf(relative: str) -> str:
    return (ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def python_selector_hash(relative: str, name: str) -> str:
    text = read_lf(relative)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(text, node)
            assert segment is not None
            return sha256(segment)
    raise KeyError(f"{name} not found in {relative}")


def rust_test_spans(relative: str) -> dict[str, str]:
    """Return {module-relative selector: span text} for every Rust test in a file."""
    lines = read_lf(relative).split("\n")
    spans: dict[str, str] = {}
    modules: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        module = MOD.match(line)
        if module:
            modules.append((len(module.group(1)), module.group(2)))
            continue
        indent = len(line) - len(line.lstrip())
        while modules and line.strip() == "}" and indent == modules[-1][0]:
            modules.pop()
        if not TEST_ATTR.match(line):
            continue
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].strip().startswith("#["):
            cursor += 1
        function = FN.match(lines[cursor])
        if not function:
            continue
        indent, name = function.groups()
        end = cursor
        while lines[end] != indent + "}":
            end += 1
        start = index
        while start > 0 and lines[start - 1].lstrip().startswith(("//", "#[")):
            start -= 1
        selector = "::".join([m for _, m in modules] + [name])
        spans[selector] = "\n".join(lines[start : end + 1]) + "\n"
    return spans


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selector_set_difference(
    bound: set[tuple[str, str]], expected: frozenset[tuple[str, str]]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (missing, extra) selector keys, each sorted, between a register and its expected list."""
    return sorted(expected - bound), sorted(bound - expected)


def exact_set_message(label: str, bound: set[tuple[str, str]], expected: frozenset[tuple[str, str]]) -> str:
    """Return "" when the sets are equal, else a message naming every missing and extra selector."""
    missing, extra = selector_set_difference(bound, expected)
    if not missing and not extra:
        return ""
    return f"{label}: missing {missing}; extra {extra}"


def bound_keys(register: dict) -> set[tuple[str, str]]:
    return {(review["test_path"], review["selector"]) for review in register["reviews"]}


# The expected selector lists are written out here, not derived from the registers,
# so deleting an entry from a register (or adding one nobody listed here) fails by
# name. Changing a list is a reviewed edit of this file.
EXPECTED_PYTHON_SELECTORS: frozenset[tuple[str, str]] = frozenset(
    {
        ("Tests/Portable/test_media_register_custody.py", "ExactSetControls.test_dropping_one_real_entry_fails_the_exact_set_check"),
        ("Tests/Portable/test_media_register_custody.py", "ExactSetControls.test_exact_set_check_names_missing_and_extra"),
        ("Tests/Portable/test_media_register_custody.py", "PythonRegisterCustody.test_every_lock_contract_test_is_bound_individually"),
        ("Tests/Portable/test_media_register_custody.py", "PythonRegisterCustody.test_every_python_binding_matches_the_working_tree"),
        ("Tests/Portable/test_media_register_custody.py", "PythonRegisterCustody.test_python_entries_claim_no_independent_review_they_lack"),
        ("Tests/Portable/test_media_register_custody.py", "PythonRegisterCustody.test_python_register_binds_exactly_the_expected_selectors"),
        ("Tests/Portable/test_media_register_custody.py", "RustRegisterCustody.test_every_rust_binding_matches_the_working_tree"),
        ("Tests/Portable/test_media_register_custody.py", "RustRegisterCustody.test_rust_entries_are_author_self_review_only"),
        ("Tests/Portable/test_media_register_custody.py", "RustRegisterCustody.test_rust_register_binds_exactly_the_expected_selectors"),
        ("Tests/Portable/test_media_worker_lock.py", "MediaWorkerLockTests.test_healthy_cancelled_acquisition_does_not_rewrite_cancellation"),
        ("Tests/Portable/test_media_worker_lock.py", "MediaWorkerLockTests.test_healthy_running_acquisition_preserves_existing_states"),
        ("Tests/Portable/test_media_worker_lock.py", "MediaWorkerLockTests.test_poisoned_acquisition_records_failure_without_relocking"),
        ("Tests/Portable/test_media_worker_lock_crlf.py", "CrlfContractTests.test_crlf_regressed_source_is_still_refused"),
        ("Tests/Portable/test_media_worker_lock_crlf.py", "CrlfContractTests.test_crlf_source_generates_the_same_probe_as_lf"),
        ("Tests/Portable/test_media_worker_lock_crlf.py", "CrlfContractTests.test_missing_build_script_message_fails_closed"),
    }
)
EXPECTED_RUST_SELECTORS: frozenset[tuple[str, str]] = frozenset(
    {
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_acl_merge_failure_fails_closed_and_leaves_the_destination_untouched"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_async_generate_no_model_loaded"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_async_generate_null_output_path"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_async_generate_null_prompt"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_async_publish_failure_never_deletes_a_file_that_took_the_staging_path"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_async_requests_are_serialized_and_all_complete"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_cancel_after_encode_before_publish_discards_staged_file"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_cancel_during_generation_leaves_destination_untouched"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_cancel_unknown_id"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_completion_publishes_full_file_and_leaves_no_partial"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_destination_appearing_after_the_check_is_not_overwritten_on_unix"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_destination_appearing_after_the_check_is_not_overwritten_on_windows"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_destination_with_an_access_acl_is_refused_and_left_untouched"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_directory_substituted_after_validation_is_not_exchanged"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_discard_staging_removes_the_generated_file_and_tolerates_absence"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_discard_staging_rescues_a_file_that_is_not_the_generated_image"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_double_init_no_panic"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_exported_load_queued_behind_generation_is_cancelled_by_shutdown"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_exported_shutdown_bumps_the_global_epoch"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_failed_load_keeps_existing_model"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_failed_permission_step_leaves_destination_untouched"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_failed_publish_step_reports_failed_and_cleans_up"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_failed_rescue_leaves_the_foreign_file_in_place_and_names_it"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_failed_restore_keeps_the_backup_and_names_its_path"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_generate_image_bytes_null_out_data"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_generate_image_bytes_null_out_len"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_generate_image_bytes_null_prompt"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_global_state_tests_take_the_serial_guard"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_inherited_default_acl_is_not_activated_by_the_mode_copy"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_init_invalid_device_returns_error"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_init_null_device_defaults_to_cuda"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_injected_preservation_failure_publishes_nothing"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_injected_replace_failure_leaves_destination_acl_and_bytes_and_removes_staging"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_link_fallback_refuses_an_occupied_destination"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_load_admitted_after_shutdown_installs_once_the_gate_frees"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_load_after_shutdown_installs_and_replacement_drops_outside_lock"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_load_model_rejects_incomplete_checkpoint_and_stays_unloaded"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_load_queued_behind_generation_is_cancelled_by_shutdown_before_reading"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_long_valid_destination_name_publishes_on_both_paths"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_metadata_error_other_than_not_found_fails_closed_on_unix"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_metadata_error_other_than_not_found_fails_closed_on_windows"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_new_destination_gets_the_directorys_inherited_acl"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_new_destination_gets_the_umask_derived_mode"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_no_replace_move_refuses_an_occupied_destination"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_noreplace_fails_closed_without_renameat2_or_hard_links"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_noreplace_falls_back_to_linking_when_renameat2_is_unsupported"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_other_regular_file_substituted_after_validation_is_left_alone"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_other_replace_errors_leave_the_destination_and_make_no_backup"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_poll_and_cancel_respond_during_generation_and_cancel_skips_queued_work"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_poll_result_unknown_id"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_publish_keeps_the_existing_destination_group"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_publish_keeps_the_existing_destination_mode_on_sync_and_async_paths"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_publish_keeps_the_existing_destinations_explicit_acl"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_replace_error_1176_with_a_backup_leaves_the_destination_alone"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_replace_error_1177_restores_the_original_from_the_backup"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_replace_file_keeps_the_original_at_the_backup_name"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_restore_does_not_overwrite_a_file_that_appeared_at_the_destination"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_shutdown_during_load_discards_the_stale_model"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_shutdown_twice_no_panic"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_shutdown_without_init_no_panic"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_slow_publish_does_not_block_status_of_other_requests"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_staged_files_inherited_acl_is_stripped_before_the_mode_is_copied"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_staging_file_acl_is_private_before_publish"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_staging_file_is_private_while_it_exists"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_staging_path_is_short_unique_and_beside_destination"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_stale_existing_decision_does_not_create_the_destination"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_successful_replace_leaves_no_backup_or_staging_file"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_symlink_destination_is_refused"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_symlink_substituted_after_validation_is_left_alone"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_sync_entry_points_queue_behind_gate_without_blocking_status"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_sync_publish_failure_never_deletes_a_file_that_took_the_staging_path"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_sync_write_failure_keeps_existing_destination_and_leaves_no_partial"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_target_removed_before_the_replace_is_not_published_as_a_replacement"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_understand_image_no_model_loaded"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_understand_image_null_image_path"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_understand_image_null_prompt"),
        ("Native/pcai_core/pcai_media/src/ffi/mod.rs", "ffi::tests::test_write_and_publish_failures_leave_no_partial_and_keep_destination"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_generation_metadata_7b_schema_preserves_required_components"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_generation_metadata_probe_uses_scalar_storage_for_7b_shape"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_generation_metadata_rejects_invalid_attention_dimensions"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_generation_weights_load_without_optional_vision"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_generation_weights_report_complete_understanding_aligner"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_generation_weights_require_generation_with_complete_aligner"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_gguf_quantization_path_routing"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_load_refuses_pytorch_only_directory_with_conversion_command"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_load_rejects_empty_generation_archive_before_allocation"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_load_rejects_missing_generation_weights_in_both_modes"),
        ("Native/pcai_core/pcai_media/src/generate.rs", "generate::tests::test_load_rejects_nonempty_partial_7b_checkpoint_before_allocation"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_cold_cache_falls_back_to_download_and_hint_names_the_commands"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_corrupt_and_truncated_safetensors_are_rejected"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_load_weights_accepts_complete_mixed_prefix_archive"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_load_weights_rejects_empty_archive"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_load_weights_rejects_partial_archive_before_mutation"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_load_weights_rejects_unregistered_variables"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_load_weights_rejects_wrong_shape"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_local_pytorch_directory_hint_converts_in_place"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_missing_explicit_path_is_an_error_not_a_prepared_fallback"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_only_exact_supported_ids_resolve_to_prepared_copy"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_preparation_commands_are_runnable_and_match_the_documentation"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_prepared_copy_resolves_without_network"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_weight_metadata_accepts_complete_mixed_prefix_headers"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_weight_metadata_rejects_empty_requirements"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_weight_metadata_rejects_partial_headers"),
        ("Native/pcai_core/pcai_media/src/hub.rs", "hub::tests::test_weight_metadata_rejects_wrong_shape_headers"),
        ("Native/pcai_core/pcai_media/src/upscale.rs", "upscale::tests::test_image_to_tensor_shape"),
        ("Native/pcai_core/pcai_media/src/upscale.rs", "upscale::tests::test_tensor_to_image_preserves_rgb_planes"),
        ("Native/pcai_core/pcai_media/src/upscale.rs", "upscale::tests::test_tensor_to_image_rejects_oversized_expected_dimensions"),
        ("Native/pcai_core/pcai_media/src/upscale.rs", "upscale::tests::test_tensor_to_image_rejects_wrong_spatial_shape"),
        ("Native/pcai_core/pcai_media_model/src/janus_llama.rs", "janus_llama::tests::test_cpu_forward_with_flash_requested"),
        ("Native/pcai_core/pcai_media_model/src/janus_llama_quantized.rs", "janus_llama_quantized::tests::test_from_gguf_smoke"),
        ("Native/pcai_core/pcai_media_model/src/lib.rs", "tests::test_janus_model_construct"),
        ("Native/pcai_core/pcai_media_model/src/lib.rs", "tests::test_janus_model_construct_1b"),
        ("Native/pcai_core/pcai_media_server/src/main.rs", "tests::test_generate_boundary_forwards_request_overrides"),
        ("Native/pcai_core/pcai_media_server/src/main.rs", "tests::test_health_response_serde"),
        ("Native/pcai_core/pcai_media_server/src/main.rs", "tests::test_inference_admission_released_on_generation_error"),
        ("Native/pcai_core/pcai_media_server/src/main.rs", "tests::test_invalid_requests_fail_before_inference_admission"),
        ("Native/pcai_core/pcai_media_server/src/main.rs", "tests::test_shared_inference_admission_preserves_health_and_releases"),
    }
)


class PythonRegisterCustody(unittest.TestCase):
    def test_every_python_binding_matches_the_working_tree(self):
        """Protects: the Python selectors in media-worker-lock-review.json stay bound to real bytes.

        Detects: a test, selector body, fixture peer or the FFI source edited
        without re-binding its hash (a stale register that still looks reviewed).
        Needs: the repository files only; no Cargo, model or device.
        """
        register = load(PYTHON_REGISTER)
        for review in register["reviews"]:
            with self.subTest(selector=review["selector"]):
                self.assertEqual(sha256(read_lf(review["test_path"])), review["test_sha256"])
                self.assertEqual(
                    python_selector_hash(review["test_path"], review["selector"].split(".")[-1]),
                    review["selector_sha256"],
                )
                self.assertEqual(sha256(read_lf(review["source_path"])), review["source_sha256"])
                for peer in review["fixture_peers"]:
                    self.assertEqual(sha256(read_lf(peer["path"])), peer["sha256"], peer["path"])

    def test_every_lock_contract_test_is_bound_individually(self):
        """Protects: no lock-contract test runs without its own register entry.

        Detects: a new test_* method added to the lock or CRLF test files that
        was never bound, which would let it count as covered without a hash.
        Needs: the repository files only.
        """
        bound = {r["selector"] for r in load(PYTHON_REGISTER)["reviews"]}
        for relative in (
            "Tests/Portable/test_media_worker_lock.py",
            "Tests/Portable/test_media_worker_lock_crlf.py",
            "Tests/Portable/test_media_register_custody.py",
        ):
            tree = ast.parse(read_lf(relative))
            for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                for method in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                    if method.name.startswith("test_"):
                        with self.subTest(selector=f"{cls.name}.{method.name}"):
                            self.assertIn(f"{cls.name}.{method.name}", bound)

    def test_python_entries_claim_no_independent_review_they_lack(self):
        """Protects: an author self-review is never recorded as an independent one.

        Detects: a verdict or reviewer type implying independent approval while
        the receipt hash is empty.
        Needs: the repository files only.
        """
        for review in load(PYTHON_REGISTER)["reviews"]:
            with self.subTest(selector=review["selector"]):
                if review["review_receipt_sha256"] is None:
                    self.assertTrue(review["review_verdict"].startswith("pending"), review["review_verdict"])
                    self.assertNotIn("independent-approved", review["review_verdict"])
                    self.assertTrue(
                        review["reviewer_type"].startswith("author"), review["reviewer_type"]
                    )

    def test_python_register_binds_exactly_the_expected_selectors(self):
        """Protects: the Python register keeps every expected selector and gains none unlisted.

        Detects: an entry deleted from media-worker-lock-review.json (a count
        threshold would let any other entry stand in for it) or an entry added
        without being listed in EXPECTED_PYTHON_SELECTORS. The failure names each one.
        Needs: the repository files only.
        """
        message = exact_set_message(
            "media-worker-lock-review.json", bound_keys(load(PYTHON_REGISTER)), EXPECTED_PYTHON_SELECTORS
        )
        self.assertEqual(message, "")


class RustRegisterCustody(unittest.TestCase):
    def test_every_rust_binding_matches_the_working_tree(self):
        """Protects: each Rust selector in media-weight-safety-review.json stays bound to its text.

        Detects: a media test or its source file changing without a re-bound hash,
        a selector that no longer exists, or a duplicate entry.
        Needs: the repository files only; Rust is parsed as text, not built.
        """
        register = load(RUST_REGISTER)
        reviews = register["reviews"]
        keys = [(r["test_path"], r["selector"]) for r in reviews]
        self.assertEqual(len(keys), len(set(keys)), "duplicate selector entries")
        cache: dict[str, dict[str, str]] = {}
        for review in reviews:
            with self.subTest(selector=review["selector"], path=review["test_path"]):
                spans = cache.setdefault(review["test_path"], rust_test_spans(review["test_path"]))
                module_less = review["selector"]
                candidates = [k for k in spans if module_less == k or module_less.endswith("::" + k)]
                self.assertEqual(len(candidates), 1, f"selector not found uniquely: {candidates}")
                self.assertEqual(sha256(spans[candidates[0]]), review["selector_sha256"])
                for source in review["source_files"]:
                    self.assertEqual(sha256(read_lf(source["path"])), source["sha256"], source["path"])

    def test_rust_entries_are_author_self_review_only(self):
        """Protects: nothing in the Rust register is presented as independently reviewed.

        Detects: reviewer metadata that claims independence, a missing rationale
        field, or a disposition other than keep without an explanation.
        Needs: the repository files only.
        """
        for review in load(RUST_REGISTER)["reviews"]:
            with self.subTest(selector=review["selector"]):
                self.assertIn("independent review pending", review["reviewer_type"])
                for field in ("protects", "detects", "needs"):
                    self.assertTrue(review[field].strip(), field)
                self.assertEqual(review["disposition"], "keep")

    def test_rust_register_binds_exactly_the_expected_selectors(self):
        """Protects: the Rust register keeps every new or changed media test and gains none unlisted.

        Detects: an entry deleted from media-weight-safety-review.json, which a
        minimum-count check would hide whenever another entry remained, or an
        unlisted addition. The failure names each missing and extra selector.
        Needs: the repository files only.
        """
        message = exact_set_message(
            "media-weight-safety-review.json", bound_keys(load(RUST_REGISTER)), EXPECTED_RUST_SELECTORS
        )
        self.assertEqual(message, "")


class ExactSetControls(unittest.TestCase):
    def test_exact_set_check_names_missing_and_extra(self):
        """Protects: the exact-set comparison reports deletions and additions by name.

        Detects: a comparison that only counts, or that ignores one direction, so
        a removed or an unlisted selector could pass.
        Needs: no files; synthetic selector sets.
        """
        expected = frozenset({("a.py", "T.one"), ("a.py", "T.two"), ("b.rs", "m::three")})
        self.assertEqual(exact_set_message("x", set(expected), expected), "")
        removed = set(expected) - {("a.py", "T.two")}
        self.assertIn("T.two", exact_set_message("x", removed, expected))
        self.assertIn("missing", exact_set_message("x", removed, expected))
        added = set(expected) | {("a.py", "T.extra")}
        self.assertIn("T.extra", exact_set_message("x", added, expected))
        self.assertIn("extra", exact_set_message("x", added, expected))
        swapped = (set(expected) - {("a.py", "T.two")}) | {("a.py", "T.other")}
        message = exact_set_message("x", swapped, expected)
        self.assertTrue("T.two" in message and "T.other" in message, message)

    def test_dropping_one_real_entry_fails_the_exact_set_check(self):
        """Protects: removing any single bound selector from either real register is caught.

        Detects: the deletion hole of a minimum-count check, shown on the real
        registers: for every entry, the set minus that entry must fail by name.
        Needs: the repository files only.
        """
        for register, expected in (
            (load(PYTHON_REGISTER), EXPECTED_PYTHON_SELECTORS),
            (load(RUST_REGISTER), EXPECTED_RUST_SELECTORS),
        ):
            keys = bound_keys(register)
            self.assertEqual(exact_set_message("x", keys, expected), "", "positive control: unmodified register")
            for key in sorted(keys):
                with self.subTest(removed=key):
                    message = exact_set_message("x", keys - {key}, expected)
                    self.assertIn(key[1], message)


if __name__ == "__main__":
    unittest.main()
