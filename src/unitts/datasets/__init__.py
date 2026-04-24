"""Benchmark text datasets for TTS evaluation."""

BENCHMARK_SENTENCES = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "The rain in Spain falls mainly on the plain.",
        "To be or not to be, that is the question.",
    ],
    "zh": [
        "今天天气真不错，我们一起出去走走吧。",
        "知识就是力量，学习永无止境。",
    ],
    "de": [
        "Die Wissenschaft hat keine moralische Dimension.",
        "Ein guter Anfang ist die halbe Arbeit.",
    ],
    "fr": [
        "La vie est belle quand on prend le temps de la savourer.",
        "Chaque pas compte sur le long chemin.",
    ],
    "tr": [
        "Bugün hava çok güzel, birlikte yürüyüşe çıkalım mı?",
        "Sabahları erken kalkmak bana iyi geliyor.",
    ],
}

# Rainbow Passage excerpt. Phonetically balanced, public domain, widely
# used in speech research for comparing synthesis quality.
LONG_TEXT_EN = (
    "When the sunlight strikes raindrops in the air, they act as a prism "
    "and form a rainbow. The rainbow is a division of white light into many "
    "beautiful colors. These take the shape of a long round arch, with its "
    "path high above, and its two ends apparently beyond the horizon."
)
