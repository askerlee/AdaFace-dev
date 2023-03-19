#!/usr/bin/fish
# " | " separated fields like "subject | prompt | folder | dreambooth-class-token".
# Temporary clear the previous cases, so as to only generate the following cases. 
# Undo clearing by commenting out the following line.
set cases
set -l cases "alita | a {} | alita | girl, instagram"
set -a cases "alita | leading an army on a holy war on Mars | alita-mars | girl, instagram"
set -a cases "alita | as a Disney Princess | alita-disney | girl, instagram"
set -Ux cases $cases
