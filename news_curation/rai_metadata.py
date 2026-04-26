"""Responsible AI metadata for the News curation.

This is the source of truth for the RAI fields that get merged into the
auto-generated Croissant metadata when `tools/meta_data.py --curation news`
is run. Edit this file when the documentation needs an update; do not edit
the generated JSON directly.

Conformance: http://mlcommons.org/croissant/RAI/1.0
"""

NEWS_RAI = {
    "rai:dataCollection": (
        "Articles are sourced from the Common Crawl News dataset (CC-NEWS), a continuous "
        "monthly archive of public news pages collected by web crawls. For each version "
        "(e.g. 2025/09), all WARC files for that month are downloaded from "
        "data.commoncrawl.org. Article text is extracted from each crawled HTML page using "
        "trafilatura (with metadata=True, no comments/links/tables) and a fast HTML-lang "
        "pre-filter. After extraction, langdetect on the first 1000 characters confirms the "
        "article is in the configured language (default: English). No human curation, no "
        "editorial filtering of sources beyond what CC-NEWS itself includes."
    ),
    "rai:dataCollection": "The dataset was constructed from CC-NEWS articles. CC-NEWS is a Common Crawl news dataset containing news articles from news sites around the world, released as WARC files in the CC-NEWS. Source articles were selected according to the documented CC-NEWS version or date range, language filters, article-length filters, deduplication criteria, and availability constraints. The source articles are real-world news text from CC-NEWS. The questions, answers, and other labels of the QAs were produced by Gemini models as machine-generated annotations.",
    "rai:dataCollectionType": [
        "Secondary Data analysis",
        "Document analysis",
        "Others: LLM-based synthetic annotation generation"
    ],
    "rai:dataCollectionRawData": "The raw source data consisted of CC-NEWS article records ( .warc.gz files from a single month of Common Crawl News), including article text and available metadata such as URL, crawl timestamp, publisher/domain, language, and any retained article identifiers. The derived dataset additionally contains machine-generated questions, reference answers, closed-book answers, open-book answers, QA-quality judgments, and answer-quality judgments. The exact CC-NEWS snapshot/date range, article selection code, clustering configuration, prompts, Gemini model settings, and filtering rules are documented in the dataset curation repository.",

    "rai:dataCollectionTimeFrameStart": {
        "@value": "2025-09-01T00:00:00",
        "dataType": "sc:Date"
    },
    "rai:dataCollectionTimeFrameEnd": {
        "@value": "2025-09-30T00:00:00",
        "dataType": "sc:Date"
    },

    "rai:dataBiases": [
        "The dataset may inherit selection and coverage biases from CC-NEWS, including overrepresentation of highly covered regions, languages, publishers, events, and public figures.",
        "The clustering process may amplify popular or repetitive news topics while underrepresenting less-covered events.",
        "Gemini-generated QAs, answers, and judgments may reflect model biases, rubric sensitivity, position effects, or preference for fluent but unsupported answers.",
        "Open-book and closed-book answers may be affected by model memorization, temporal leakage, or prior knowledge of widely reported events."
    ],

    "rai:dataLimitations": [
        "The source articles are real-world news articles derived from CC-NEWS, while the questions, answers, and judgments are machine-generated using Gemini.",
        "Generated questions and answers may contain hallucinations, ambiguous wording, incomplete grounding, or factual errors.",
        "Machine-judge labels should not be treated as infallible human-verified ground truth.",
        "The dataset reflects the topical, geographic, linguistic, temporal, and editorial coverage biases of CC-NEWS and the behavior of the Gemini models used in the pipeline.",
        "The dataset is not recommended for making factual claims about current events, evaluating real people, legal or medical decision-making, or other high-stakes applications."
    ],

    "rai:personalSensitiveInformation": [
        "The dataset may contain references to real people, public figures, organizations, locations, crimes, disasters, health issues, political views, legal allegations, or other sensitive news topics because it is derived from news articles.",
        "No new personal data was intentionally collected from individuals by the dataset creators.",
        "No PII removal is performed beyond what trafilatura's HTML "
        "extraction already strips. Users training models on this dataset should consider "
        "the implications: model outputs may reproduce names, quotes, and other identifying "
        "details verbatim. The dataset is derived from publicly accessible news pages. It does not include "
        "scraped private content, login-required pages, or social media posts beyond what "
        "is reproduced inside the news articles themselves."    
    ],

    "rai:dataUseCases": [
        "Benchmarking language models on factual question answering grounded in real news "
        "events.",
        "Recommended uses include research on synthetic-data, machine unlearning, memorization, retrieval augmentation, etc.",
        "Not recommended: making real-world decisions about individuals, organizations, or "
        "events mentioned in the articles. The dataset is for evaluation, not as ground-truth "
        "reporting.",
    ],
    
    "rai:dataSocialImpact": "This dataset may support research on open-book versus closed-book question answering, retrieval-augmented generation, factuality evaluation, news-grounded QA, and LLM-as-judge reliability. Positive impacts include enabling more systematic study of whether models answer from evidence, from parametric knowledge, or from hallucination. Risks include reproducing biases and harms present in news coverage, including overrepresentation of highly covered regions, publishers, languages, public figures, conflicts, crimes, disasters, and political events. Because the source data is news, examples may mention real people, organizations, locations, allegations, health events, political views, religion, or other sensitive topics. Additional risks include treating Gemini-generated questions, answers, or judge labels as human-verified ground truth; using the dataset to make factual claims about current events; amplifying misinformation or outdated news; and evaluating models in ways that disadvantage under-covered communities or languages. Mitigations include documenting the CC-NEWS source range, retaining source identifiers, disclosing all Gemini generation and judging stages, filtering malformed or low-quality examples, screening for sensitive or harmful content where applicable, reporting known limitations of machine judging, and discouraging use for high-stakes decisions about real people or current events.",

    "rai:hasSyntheticData": True,
    
    "prov:wasGeneratedBy": [
        "Source data acquisition: The dataset was derived from CC-NEWS using the documented corpus version, date range, language filters, and article selection criteria.",
        "Article preprocessing: Source articles were parsed, cleaned, filtered, deduplicated where applicable, and converted into the internal schema used by the QA pipeline.",
        "Article clustering: Preprocessed articles were clustered into topically related groups using the documented embedding, similarity, and clustering procedure.",
        "Question-answer generation: Gemini generated QA pairs from clustered article sets using documented prompt templates, model settings, and parsing logic.",
        "Closed-book answer generation: Gemini generated answers to the questions without access to the source articles, according to the documented closed-book prompt and model settings.",
        "Open-book answer generation: Gemini generated answers to the questions with access to relevant article evidence or clustered article context, according to the documented open-book prompt and model settings.",
        "QA quality judging: Gemini judged generated QA pairs for answerability, relevance, clarity, factual consistency, ambiguity, and suitability using the documented rubric.",
        "Open-book and closed-book answer judging: Gemini judged the generated open-book and closed-book answers against the question, evidence where available, and the documented scoring rubric.",
        "Filtering and dataset construction: Records were filtered using schema validation, duplicate detection, quality thresholds, judge scores, and any documented automated or manual review.",
        "Responsible AI review: The dataset was reviewed for source attribution, licensing and redistribution constraints, sensitive real-world news content, privacy risks, harmful content, and limitations of LLM-generated answers and judgments.",
        "Dataset packaging and release: The final dataset was packaged with generated QAs, open-book answers, closed-book answers, judge outputs, filtering metadata, Croissant metadata, and reproducibility materials. Full prompts, scripts, model settings, clustering details, and quality-control procedures are available in the dataset curation repository: https://github.com/plau666/ContinuousBenchCuration/tree/master/news_curation."
    ],
    "prov:wasDerivedFrom": [
        "CC-NEWS corpus: The dataset was derived from CC-NEWS in https://data.commoncrawl.org/crawl-data/CC-NEWS/index.html"
        "Dataset curation repository with prompts, clustering code, generation scripts, model settings, filtering rules, and reproducibility materials: https://github.com/plau666/ContinuousBenchCuration/tree/master/news_curation"
    ]
}
