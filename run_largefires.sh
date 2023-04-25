#!/bin/bash
mkdir output

# set -euxo pipefail
set -eo pipefail

basedir=$( cd "$(dirname "$0")"; pwd -P )
echo "Basedir: $basedir"
echo "Initial working directory: $(pwd -P)"

echo "conda: $(which conda)"

# Only create conda environment if it's not currently active. This allows for
# interactive testing of this script.
if [[ $CONDA_PREFIX != "/projects/env-feds" ]]; then
  # Trying to resolve conda permissiosn issue
  # https://gitter.im/conda/conda?at=5dc427aa2f8a034357513172
  export CONDA_PKGS_DIRS="$basedir/.conda"
  mkdir -p "$CONDA_PKGS_DIRS"
  conda env create -f "$basedir/env-feds.yml" -p "$basedir/env-feds"
  source activate "$basedir/env-feds"
fi

echo "Python: $(which python)"
python --version

echo "Starting algorithm in subshell"
(
pushd "$basedir"
{ # try
  echo "Running in directory: $(pwd -P)"
  # run the ps command every second in the background with `watch` command
  #cmd="ps -o pid,user,%mem,command ax | sort -b -k3 -r  | grep python >> /app/fireatlas_nrt/running.log"
  #watch -n 1 $cmd &
  #bash run_ps_memory.sh &
  python combine_largefire.py -s 2023 -e 2023 -p -x
  popd
  echo "Copying log to special output dir"
  cp "$basedir/running.log" ./output
  cp "$basedir/memory.log" ./output
} || { # catch
  popd
  echo "Copying log to special output dir"
  cp "$basedir/running.log" ./output
}
)
echo "Done!"

exit