#!/usr/bin/fish
# " | " separated fields like "subject | prompt | folder | dreambooth-class-token".
# Temporary clear the previous cases, so as to only generate the following cases. 
# Undo clearing by commenting out the following line.
set -a cases "taylorswift | a reflexing water a cute sad {} half submerged in the lake water just the eyes and head above water, glares and reflections like in a mirror, depth of field, portrait, kodak portra 400, film grain and nice chromatic bokeh, 105mm f1.4 | taylorswift-water | girl, instagram"
set -a cases "iainarmitage | as a crochet figure | iainarmitage-crochet | young boy, instagram"
set -Ux cases $cases
