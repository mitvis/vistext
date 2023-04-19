def cleanL2L3(caption):
    caption = caption.strip()
    if caption[-1] != ".":
        caption = caption+"."
    return caption