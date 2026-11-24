#!/bin/bash
# ---------------------------------------------------------------------
#  Rename the CroppedYale subject folders so they are a simple numeric
#  range, the same shape as ORL:
#
#      yaleB01, yaleB02, ..., yaleB13, yaleB15, ..., yaleB39
#   -> s1,      s2,      ..., s13,     s14,     ..., s38
#
#  WHY.  The original names are not contiguous (yaleB14 does not exist),
#  so the MATLAB code had to list the directory and filter it.  After this
#  rename the code can just count 1..38 and build the path, exactly like
#  it already does for ORL.
#
#  IS THIS SAFE FOR THE RESULTS?  Yes.  The labels in the MATLAB code were
#  never taken from the folder name -- they are the loop index -- and the
#  alphabetical order of yaleB01..yaleB39 is the same order as s1..s38, so
#  every image ends up in the same class as before.  Recognition rates are
#  unchanged.
#
#  USAGE
#      ./rename_yale.sh          rename yaleB* -> s*
#      ./rename_yale.sh --undo   put the original names back
#
#  The mapping is written to yale_rename_map.txt so the rename is always
#  reversible, even if you forget what it did.
# ---------------------------------------------------------------------

set -e                                  # stop on the first error

# Find CroppedYale relative to this script, not the current folder.
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
YALE="$SCRIPT_DIR/../CroppedYale"
MAP="$SCRIPT_DIR/yale_rename_map.txt"

if [ ! -d "$YALE" ]; then
    echo "ERROR: cannot find $YALE"
    exit 1
fi

# ---------------- undo ----------------
if [ "$1" = "--undo" ]; then
    if [ ! -f "$MAP" ]; then
        echo "ERROR: no $MAP, so there is nothing to undo."
        exit 1
    fi
    echo "Restoring the original folder names..."
    # Read the map backwards (new -> old) and rename back.
    while read -r old new; do
        if [ -d "$YALE/$new" ]; then
            mv "$YALE/$new" "$YALE/$old"
            echo "  $new -> $old"
        fi
    done < "$MAP"
    rm "$MAP"
    echo "Done. The folders are back to yaleB* names."
    exit 0
fi

# ---------------- rename ----------------
if [ -f "$MAP" ]; then
    echo "ERROR: $MAP already exists, so the rename looks done already."
    echo "       Run './rename_yale.sh --undo' first if you want to redo it."
    exit 1
fi

echo "Renaming subject folders in $YALE"
: > "$MAP"                              # create or empty the map file

n=0
for folder in "$YALE"/yaleB*; do
    [ -d "$folder" ] || continue        # skip anything that is not a folder
    old=$(basename "$folder")
    n=$((n + 1))
    new="s$n"
    mv "$folder" "$YALE/$new"
    echo "$old $new" >> "$MAP"
    echo "  $old -> $new"
done

echo "Renamed $n folders. Mapping saved in $MAP"
echo "Run './rename_yale.sh --undo' to reverse it."
