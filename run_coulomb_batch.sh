#!/usr/bin/env bash


script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"

if [[ ! -d "${build_dir}" ]]; then
  echo "Build directory not found: ${build_dir}" >&2
  echo "Configure the project with CMake before running this script." >&2
  exit 1
fi

echo "Building steady_state..."
cmake --build "${build_dir}" --target steady_state --parallel

mapfile -t input_files < <(
  find "${script_dir}" -maxdepth 1 -type f \
    -name 'ras_ss_coulomb_*.in' -print | sort -V
)

if (( ${#input_files[@]} == 0 )); then
  echo "No ras_ss_coulomb_*.in input files found in ${script_dir}." >&2
  exit 1
fi

for input_file in "${input_files[@]}"; do
  echo
  echo "Running $(basename "${input_file}")..."
  (
    cd "${build_dir}"
    ./steady_state "${input_file}"
  )
done

echo
echo "Completed ${#input_files[@]} Coulomb simulations."
