#!/bin/bash

set -e

# Check for required commands
command -v bindfs >/dev/null 2>&1 || { echo >&2 "bindfs is required but it's not installed. Aborting."; exit 1; }
command -v fakechroot >/dev/null 2>&1 || { echo >&2 "fakechroot is required but it's not installed. Aborting."; exit 1; }
command -v fusermount >/dev/null 2>&1 || { echo >&2 "fusermount is required but it's not installed. Aborting."; exit 1; }

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 conf-file command [args...]"
    exit 1
fi

CONF_FILE="$1"
COMMAND="$2"
shift 2
ARGS=("$@")

if [ ! -f "$CONF_FILE" ]; then
    echo "Configuration file $CONF_FILE does not exist."
    exit 1
fi

WORKDIR=$(mktemp -d)
MOUNT_POINTS=()

cleanup() {
    echo "Cleaning up..."
    for mount_point in "${MOUNT_POINTS[@]}"; do
        echo "Unmounting $mount_point"
        fusermount -u "$mount_point" || true
    done
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

# Copy files and bind directories as per conf-file
IFS=$'\n'
for line in $(cat "$CONF_FILE"); do
    ORIGIN=$(echo "$line" | awk '{print $1}')
    DEST=$(echo "$line" | awk '{print $2}')
    DEST_DIR="$WORKDIR$(dirname "$DEST")"
    mkdir -p "$DEST_DIR"

    if [ -d "$ORIGIN" ]; then
        echo "Binding directory $ORIGIN to $WORKDIR$DEST"
        mkdir -p "$WORKDIR$DEST"  # Ensure the destination directory exists
        bindfs --no-allow-other "$ORIGIN" "$WORKDIR$DEST"
        MOUNT_POINTS+=("$WORKDIR$DEST")
    else
        echo "Copying file $ORIGIN to $WORKDIR$DEST"
        cp "$ORIGIN" "$WORKDIR$DEST"
    fi
done

# Run the command in the fake chroot environment
echo "Executing command: fakechroot chroot $WORKDIR $COMMAND ${ARGS[*]}"
fakechroot chroot "$WORKDIR" "$COMMAND" "${ARGS[@]}"

