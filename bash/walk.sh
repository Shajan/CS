# Script to perform actions across folders
#
#  Place this script in all folders where you need an action performed
#    All scripts need to have the same name - including one in the root folder
#    Select pre-order and/or post-order processing at each level by modifying the script
#
#  Reccursively performs action one folder deep
#    skips folders (and it's children) that don't have this script
#
#
#  Initial commandline argument is passed down, modify this behavior if necessary

SCRIPT=$(basename "$0")

# For pre-order processing (things that need do be done before working on folders)
# Do something interesting here

for folder in *
do
  if [[ -d "$folder" && -f "$folder/$SCRIPT" ]]
  then
    pushd $folder
    ./$SCRIPT "$*"
    popd
  fi
done

# For post-order processing (things that need to be done after processing folders)
# Do something interesting here
