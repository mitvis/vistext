import json
import datasets
import os

logger = datasets.logging.get_logger(__name__)

### TO UPDATE
_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

### TO UPDATE
_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

### TO UPDATE
_HOMEPAGE = ""

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
                processed.append({"id": datum["caption_id"]+"_L1",
                                  "scenegraph": datum["scenegraph"],
                                  "datatable": datum["datatable"],
                                  "caption": "translate Chart to L1: " + datum['caption_L1'].strip()})
                processed.append({"id": datum["caption_id"]+"_L2L3",
                                  "scenegraph": datum["scenegraph"],
                                  "datatable": datum["datatable"],
                                  "caption": "translate Chart to L2L3: " + datum['caption_L2L3'].strip()})
                
            for item in processed:
                yield item["id"], {
                    "id": item["id"],
                    "scenegraph": item["scenegraph"],
                    "datatable": item["datatable"],
                    "caption": item['caption']}