"""Benchmark text datasets for TTS evaluation."""

# Standard benchmark sentences
BENCHMARK_SENTENCES = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "The rain in Spain falls mainly on the plain.",
        "To be or not to be, that is the question.",
        "I have a dream that one day this nation will rise up.",
        "Technology is best when it brings people together.",
        "The only way to do great work is to love what you do.",
        "In the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
        "The future belongs to those who believe in the beauty of their dreams.",
    ],
    "zh": [
        "今天天气真不错，我们一起出去走走吧。",
        "人工智能正在改变我们的生活方式。",
        "知识就是力量，学习永无止境。",
    ],
    "ja": [
        "今日はとても良い天気ですね。散歩に行きましょう。",
        "人工知能は私たちの生活を変えています。",
    ],
    "de": [
        "Die Wissenschaft hat keine moralische Dimension.",
        "Alle Menschen werden Bruder, wo dein sanfter Flugel weilt.",
    ],
    "fr": [
        "La vie est belle quand on prend le temps de la savourer.",
        "L'intelligence artificielle transforme notre monde.",
    ],
    "tr": [
        "Bugün hava çok güzel, birlikte yürüyüşe çıkalım mı?",
        "Yapay zeka hayatımızı değiştiriyor.",
    ],
}

LONG_TEXT_EN = (
    "Artificial intelligence has made remarkable progress in the field of speech synthesis. "
    "Modern text-to-speech systems can now produce speech that is nearly indistinguishable "
    "from human recordings. This advancement has been driven by deep learning architectures "
    "such as transformers, diffusion models, and flow matching techniques. Voice cloning "
    "technology now allows these systems to replicate any voice from just a few seconds of "
    "reference audio. The implications for accessibility, content creation, and human-computer "
    "interaction are profound."
)
