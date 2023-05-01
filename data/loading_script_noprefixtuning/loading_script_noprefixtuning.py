import json
import datasets
import os

logger = datasets.logging.get_logger(__name__)

### TO UPDATE
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

### TO UPDATE
_DESCRIPTION = """\
...
"""

### TO UPDATE
_HOMEPAGE = ""

### TO UPDATE
_LICENSE = ""

class VisTextConfig(datasets.BuilderConfig):
    """BuilderConfig for VisText."""
    
    def __init__(self, **kwargs):
        """BuilderConfig for VisText.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(VisTextConfig, self).__init__(**kwargs)
    
class VisText(datasets.GeneratorBasedBuilder):
    """VisText: A Benchmark for Semantically Rich Chart Captioning. Version 1.0."""

    BUILDER_CONFIGS = [
        VisTextConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "scenegraph": datasets.Value("string"),
                    "datatable": datasets.Value("string"),
                    "caption": datasets.Value("string")
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )
    
    def _split_generators(self, dl_manager: datasets.DownloadManager):
        files = dl_manager.download_and_extract(self.config.data_files)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": files["train"][0]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": files["validation"][0]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": files["test"][0]}),
        ]
    
    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, encoding="utf-8") as f:
            vistext = json.load(f)
            processed = []
            for datum in vistext:
                processed.append({"id": datum["caption_id"],
                                  "scenegraph": datum["scenegraph"],
                                  "datatable": datum["datatable"],
                                  "caption": datum['caption_L1'].strip() + " " + datum['caption_L2L3'].strip()})
                
            for item in processed:
                yield item["id"], {
                    "id": item["id"],
                    "scenegraph": item["scenegraph"],
                    "datatable": item["datatable"],
                    "caption": item['caption']}