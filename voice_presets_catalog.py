"""Voice preset catalog for OmniVoice FastAPI service.

Preset audio files live in the zipvoice voice_presets directory.
Each preset supplies a ref_audio WAV and matching ref_text transcript,
replacing any user-uploaded reference audio when a preset is selected.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException

_DEFAULT_PRESETS_DIR = Path(__file__).resolve().parent / "voice_presets"

PRESETS_DIR = Path(os.getenv("VOICE_PRESETS_DIR", str(_DEFAULT_PRESETS_DIR)))

# OmniVoice language display name (any case) → preset directory language code
_LANG_TO_CODE: Dict[str, str] = {
    "english": "en",
    "turkish": "tr",
    "chinese": "zh",
    "standard arabic": "ar",
    "arabic": "ar",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "russian": "ru",
    "portuguese": "pt",
    "hungarian": "hu",
    "japanese": "ja",
    "polish": "pl",
}

# Catalog: lang_code → preset_id → {name, emotions: {emotion_id → {file, text}}}
# Mirrors zipvoice VOICE_PRESETS; kept here so OmniVoice has no runtime
# dependency on the zipvoice codebase.
PRESET_CATALOG: Dict[str, Dict] = {
    "tr": {
        "man_voice_ballad": {
            "name": "Burak",
            "emotions": {
                "angry":    {"file": "man_angry.wav",    "text": "Sen kendini ne sanıyorsun? Beni kızdırma!"},
                "happy":    {"file": "man_happy.wav",    "text": "Ne kadar güzel bir gün. Çok mutluyum."},
                "sad":      {"file": "man_sad.wav",      "text": "Bugün çok mutsuz hissediyorum. Lütfen beni yalnız bırak."},
                "surprised":{"file": "man_surprised.wav","text": "Nerden çıktın sen? Seni gördüğüme çok şaşırdım!"},
            },
        },
        "woman_voice_coral": {
            "name": "Derya",
            "emotions": {
                "angry":    {"file": "woman_angry.wav",    "text": "Sen kendini ne sanıyorsun? Ben çok sinirliyim."},
                "happy":    {"file": "woman_happy.wav",    "text": "Mutluluktan havalara uçucam. Çok mutluyum."},
                "sad":      {"file": "woman_sad.wav",      "text": "Bugün çok mutsuz hissediyorum. Beni yalnız bırak."},
                "surprised":{"file": "woman_surprised.wav","text": "Seni gördüğüme çok şaşırdım. Hiç beklemiyordum seni görmeyi."},
            },
        },
        "woman_voice_call_center": {
            "name": "Melis",
            "emotions": {
                "angry":     {"file": "woman_angry.wav",   "text": "Beni dinlemiyorsunuz. Sürekli aynı şeyi tekrar etmek zorunda kalıyorum."},
                "happy":     {"file": "woman_happy.wav",   "text": "Size yardımcı olabildiğim için gerçekten çok memnun oldum."},
                "sad":       {"file": "woman_sad.wav",     "text": "Yaşadığınız bu aksaklıktan dolayı üzgünüm."},
                "surprised": {"file": "woman_surprised.wav","text": "Talebinizin olumlu sonuçlandığını sizinle paylaşmaktan büyük bir heyecan duyuyorum."},
                "empathetic":{"file": "woman_empathy.wav", "text": "Evet, yaşadığınız bu durumun zor olduğunu anlıyorum."},
            },
        },
        "child_voice_lively_girl": {
            "name": "Ceren",
            "emotions": {
                "angry":    {"file": "child_angry.wav",    "text": "Beni sinirlendirme! Ben çok sinirli birisiyim."},
                "happy":    {"file": "child_happy.wav",    "text": "Çok mutluyum. Ne kadar güzel, güneşli bir gün."},
                "sad":      {"file": "child_sad.wav",      "text": "Arkadaşlarımla gidemediğim için çok üzgünüm."},
                "surprised":{"file": "child_surprised.wav","text": "Şaşkınlık içerisindeyim. Hiç beklemiyordum bunu."},
            },
        },
    },

    "en": {
        "man_voice_deep": {
            "name": "David",
            "emotions": {
                "angry":    {"file": "en_man_angry.wav",    "text": "I told you already. Don't make me say it again."},
                "happy":    {"file": "en_man_happy.wav",    "text": "What a wonderful day. I'm feeling absolutely fantastic."},
                "sad":      {"file": "en_man_sad.wav",      "text": "I'm feeling quite down today. Please leave me alone."},
                "surprised":{"file": "en_man_surprised.wav","text": "I wasn't expecting to see you here at all. How surprised."},
            },
        },
        "woman_voice_wise": {
            "name": "Sarah",
            "emotions": {
                "angry":    {"file": "en_woman_angry.wav",    "text": "I'm getting really frustrated with this situation."},
                "happy":    {"file": "en_woman_happy.wav",    "text": "I'm absolutely delighted and couldn't be happier."},
                "sad":      {"file": "en_woman_sad.wav",      "text": "I'm feeling quite melancholy today. I need some time alone."},
                "surprised":{"file": "en_woman_surprised.wav","text": "Oh my goodness. What a pleasant surprise to see you."},
            },
        },
        "child_voice_lovely": {
            "name": "Laura",
            "emotions": {
                "angry":    {"file": "en_child_angry.wav",    "text": "That's not fair. You said we'd play, but now you don't want to."},
                "happy":    {"file": "en_child_happy.wav",    "text": "I get to play with my friends and eat ice cream."},
                "sad":      {"file": "en_child_sad.wav",      "text": "I lost my toy... now I feel really sad."},
                "surprised":{"file": "en_child_surprised.wav","text": "I can't wait to see what's inside the box. So excited."},
            },
        },
    },

    "zh": {
        "man_voice_standard": {
            "name": "李明",
            "emotions": {
                "angry":    {"file": "zh_man_angry.wav",    "text": "我真的不喜欢这样！"},
                "happy":    {"file": "zh_man_happy.wav",    "text": "我今天很开心！"},
                "sad":      {"file": "zh_man_sad.wav",      "text": "我怀念过去的样子。"},
                "surprised":{"file": "zh_man_surprised.wav","text": "真的？我不敢相信！"},
            },
        },
        "woman_voice_gentle": {
            "name": "王美丽",
            "emotions": {
                "angry":    {"file": "zh_woman_angry.wav",    "text": "我真的不喜欢这样！"},
                "happy":    {"file": "zh_woman_happy.wav",    "text": "我今天很开心！"},
                "sad":      {"file": "zh_woman_sad.wav",      "text": "我怀念过去的样子。"},
                "surprised":{"file": "zh_woman_surprised.wav","text": "真的？我不敢相信！"},
            },
        },
        "child_voice_pretty": {
            "name": "乐乐",
            "emotions": {
                "angry":    {"file": "zh_child_angry.wav",    "text": "你都没有听我说话！我说了好几次啦，你怎么还不明白！"},
                "happy":    {"file": "zh_child_happy.wav",    "text": "今天我特别开心！因为老师表扬了我，还给了我一颗糖！"},
                "sad":      {"file": "zh_child_sad.wav",      "text": "我有点难过……因为我的玩具坏掉了。妈妈，你能帮我修好吗？"},
                "surprised":{"file": "zh_child_surprised.wav","text": "哇！真的是你吗？我没想到你会来！好惊喜啊！"},
            },
        },
    },

    "ar": {
        "man_voice_formal": {
            "name": "أحمد",
            "emotions": {
                "angry":    {"file": "ar_man_angry.wav",    "text": "أنا غاضب لأنك لم تستمع لي أبدًا!"},
                "happy":    {"file": "ar_man_happy.wav",    "text": "أنا سعيد جدًا اليوم، أشعر أن كل شيء يسير على ما يرام."},
                "sad":      {"file": "ar_man_sad.wav",      "text": "أنا حزين جدًا… ولا أستطيع التحمل ذلك."},
                "surprised":{"file": "ar_man_surprised.wav","text": "لا أصدق! أنا متحمس جدا، لا أستطيع الانتظار."},
            },
        },
        "woman_voice_calm": {
            "name": "ليلى",
            "emotions": {
                "angry":    {"file": "ar_woman_angry.wav",    "text": "أنا غاضب لأنك لم تستمع لي أبدًا!"},
                "happy":    {"file": "ar_woman_happy.wav",    "text": "أنا سعيد جدًا اليوم، أشعر أن كل شيء يسير على ما يرام."},
                "sad":      {"file": "ar_woman_sad.wav",      "text": "أنا حزين جدًا… ولا أستطيع التحمل ذلك."},
                "surprised":{"file": "ar_woman_surprised.wav","text": "لا أصدق! أنا متحمس جدا، لا أستطيع الانتظار."},
            },
        },
        "child_voice_lovely": {
            "name": "مريم",
            "emotions": {
                "angry":    {"file": "ar_child_angry.wav",    "text": "أنت لا تسمعني أبداً. قلت لك هذا أكثر من مرة، لماذا لا تفهم؟"},
                "happy":    {"file": "ar_child_happy.wav",    "text": "اليوم أنا سعيدة جداً. لأن المعلمة مدحتني وأعطتني حلوى."},
                "sad":      {"file": "ar_child_sad.wav",      "text": "أنا حزينة قليلاً. لأن لعبتي انكسرت. ماما، هل يمكنك إصلاحها لي؟"},
                "surprised":{"file": "ar_child_surprised.wav","text": "واو. حقاً أنت هنا؟ لم أكن أتوقع أن تأتي! يا لها من مفاجأة جميلة."},
            },
        },
    },

    "de": {
        "man_voice_determined": {
            "name": "Lukas",
            "emotions": {
                "angry":    {"file": "de_man_angry.wav",    "text": "Ich bin richtig sauer, weil niemand auf mich hört!"},
                "happy":    {"file": "de_man_happy.wav",    "text": "Was für ein großartiger Tag. Ich fühle mich richtig fantastisch!"},
                "sad":      {"file": "de_man_sad.wav",      "text": "Es tut weh, wenn Menschen Dinge sagen, die mich traurig machen."},
                "surprised":{"file": "de_man_surprised.wav","text": "Oh mein Gott, das ist so aufregend, ich kann kaum still sitzen!"},
            },
        },
        "woman_voice_casual": {
            "name": "Anna",
            "emotions": {
                "angry":    {"file": "de_woman_angry.wav",    "text": "Ich bin richtig sauer, weil niemand auf mich hört!"},
                "happy":    {"file": "de_woman_happy.wav",    "text": "Was für ein großartiger Tag. Ich fühle mich richtig fantastisch!"},
                "sad":      {"file": "de_woman_sad.wav",      "text": "Es tut weh, wenn Menschen Dinge sagen, die mich traurig machen."},
                "surprised":{"file": "de_woman_surprised.wav","text": "Oh mein Gott, das ist so aufregend, ich kann kaum still sitzen!"},
            },
        },
        "child_voice_adorable": {
            "name": "Helga",
            "emotions": {
                "angry":    {"file": "de_child_angry.wav",    "text": "Das ist gemein! Du hast gesagt, wir spielen, und jetzt willst du nicht mehr."},
                "happy":    {"file": "de_child_happy.wav",    "text": "Ich darf mit meinen Freunden spielen und bekomme Eis! Juhu!"},
                "sad":      {"file": "de_child_sad.wav",      "text": "Ich habe mein Lieblingsspielzeug verloren... Jetzt bin ich ganz traurig."},
                "surprised":{"file": "de_child_surprised.wav","text": "Wow! Ich bin so gespannt, was in der Box ist!"},
            },
        },
    },

    "fr": {
        "man_elegant": {
            "name": "Louis",
            "emotions": {
                "angry":    {"file": "fr_man_angry.wav",    "text": "Ça suffit maintenant. Je ne peux plus supporter ça!"},
                "happy":    {"file": "fr_man_happy.wav",    "text": "Je suis tellement heureux aujourd'hui!"},
                "sad":      {"file": "fr_man_sad.wav",      "text": "Je ne sais plus quoi faire. Je suis vraiment triste."},
                "surprised":{"file": "fr_man_surprised.wav","text": "C'est incroyable! Je suis tellement excitée!"},
            },
        },
        "woman_casual": {
            "name": "Claire",
            "emotions": {
                "angry":    {"file": "fr_woman_angry.wav",    "text": "Ça suffit maintenant. Je ne peux plus supporter ça!"},
                "happy":    {"file": "fr_woman_happy.wav",    "text": "Je suis tellement heureux aujourd'hui!"},
                "sad":      {"file": "fr_woman_sad.wav",      "text": "Je ne sais plus quoi faire. Je suis vraiment triste."},
                "surprised":{"file": "fr_woman_surprised.wav","text": "C'est incroyable! Je suis tellement excitée!"},
            },
        },
        "child_lovely": {
            "name": "Eloise",
            "emotions": {
                "angry":    {"file": "fr_child_angry.wav",    "text": "Ce n'est pas juste ! Tu avais dit qu'on allait jouer, et maintenant tu ne veux plus."},
                "happy":    {"file": "fr_child_happy.wav",    "text": "Je vais jouer avec mes amis et manger de la glace !"},
                "sad":      {"file": "fr_child_sad.wav",      "text": "J'ai perdu mon jouet... maintenant je suis vraiment triste."},
                "surprised":{"file": "fr_child_surprised.wav","text": "J'ai trop hâte de voir ce qu'il y a dans la boîte. Quelle excitation !"},
            },
        },
    },

    "es": {
        "man_determined": {
            "name": "Andres",
            "emotions": {
                "angry":    {"file": "es_man_angry.wav",    "text": "¡Esto no es justo! Ya estoy cansada de esta situación."},
                "happy":    {"file": "es_man_happy.wav",    "text": "¡Estoy muy feliz hoy! Todo está saliendo mejor de lo que esperaba."},
                "sad":      {"file": "es_man_sad.wav",      "text": "Me siento muy triste… ojalá las cosas fueran diferentes."},
                "surprised":{"file": "es_man_surprised.wav","text": "¡No puedo creerlo, esto es increíble! Estoy tan emocionada."},
            },
        },
        "woman_elegant": {
            "name": "Sofia",
            "emotions": {
                "angry":    {"file": "es_woman_angry.wav",    "text": "¡Esto no es justo! Ya estoy cansada de esta situación."},
                "happy":    {"file": "es_woman_happy.wav",    "text": "¡Estoy muy feliz hoy! Todo está saliendo mejor de lo que esperaba."},
                "sad":      {"file": "es_woman_sad.wav",      "text": "Me siento muy triste… ojalá las cosas fueran diferentes."},
                "surprised":{"file": "es_woman_surprised.wav","text": "¡No puedo creerlo, esto es increíble! Estoy tan emocionada."},
            },
        },
    },

    "it": {
        "man_passionate": {
            "name": "Marco",
            "emotions": {
                "angry":    {"file": "it_man_angry.wav",    "text": "Non posso più sopportare questa situazione! Sono davvero arrabbiato!"},
                "happy":    {"file": "it_man_happy.wav",    "text": "Che giornata meravigliosa! Sono così felice oggi!"},
                "sad":      {"file": "it_man_sad.wav",      "text": "Mi sento molto triste... vorrei che le cose fossero diverse."},
                "surprised":{"file": "it_man_surprised.wav","text": "Non ci posso credere! Che sorpresa incredibile!"},
            },
        },
        "woman_elegant": {
            "name": "Giulia",
            "emotions": {
                "angry":    {"file": "it_woman_angry.wav",    "text": "Basta! Non ne posso più di questa situazione!"},
                "happy":    {"file": "it_woman_happy.wav",    "text": "Sono così felice! Tutto sta andando meravigliosamente!"},
                "sad":      {"file": "it_woman_sad.wav",      "text": "Mi sento così triste oggi... ho bisogno di stare da sola."},
                "surprised":{"file": "it_woman_surprised.wav","text": "Oh mio Dio! Che meravigliosa sorpresa!"},
            },
        },
    },

    "ru": {
        "man_strong": {
            "name": "Дмитрий",
            "emotions": {
                "angry":    {"file": "ru_man_angry.wav",    "text": "Я очень зол! Почему никто меня не слушает?"},
                "happy":    {"file": "ru_man_happy.wav",    "text": "Какой прекрасный день! Я очень счастлив сегодня!"},
                "sad":      {"file": "ru_man_sad.wav",      "text": "Мне очень грустно... Я хочу побыть один."},
                "surprised":{"file": "ru_man_surprised.wav","text": "Не могу поверить! Какой невероятный сюрприз!"},
            },
        },
        "woman_gentle": {
            "name": "Анастасия",
            "emotions": {
                "angry":    {"file": "ru_woman_angry.wav",    "text": "Хватит! Я больше не могу это терпеть!"},
                "happy":    {"file": "ru_woman_happy.wav",    "text": "Я так счастлива! Всё идёт просто замечательно!"},
                "sad":      {"file": "ru_woman_sad.wav",      "text": "Мне так грустно сегодня... Мне нужно побыть одной."},
                "surprised":{"file": "ru_woman_surprised.wav","text": "О боже! Какой чудесный сюрприз!"},
            },
        },
    },

    "pt": {
        "man_confident": {
            "name": "João",
            "emotions": {
                "angry":    {"file": "pt_man_angry.wav",    "text": "Já chega! Estou farto desta situação!"},
                "happy":    {"file": "pt_man_happy.wav",    "text": "Que dia maravilhoso! Estou muito feliz hoje!"},
                "sad":      {"file": "pt_man_sad.wav",      "text": "Estou muito triste... Preciso de ficar sozinho."},
                "surprised":{"file": "pt_man_surprised.wav","text": "Não acredito! Que surpresa incrível!"},
            },
        },
        "woman_warm": {
            "name": "Maria",
            "emotions": {
                "angry":    {"file": "pt_woman_angry.wav",    "text": "Basta! Não aguento mais esta situação!"},
                "happy":    {"file": "pt_woman_happy.wav",    "text": "Estou tão feliz! Tudo está a correr maravilhosamente!"},
                "sad":      {"file": "pt_woman_sad.wav",      "text": "Estou tão triste hoje... Preciso de estar sozinha."},
                "surprised":{"file": "pt_woman_surprised.wav","text": "Meu Deus! Que surpresa maravilhosa!"},
            },
        },
    },

    "hu": {
        "man_resolute": {
            "name": "László",
            "emotions": {
                "angry":    {"file": "hu_man_angry.wav",    "text": "Elég volt! Nagyon dühös vagyok emiatt!"},
                "happy":    {"file": "hu_man_happy.wav",    "text": "Milyen csodálatos nap! Nagyon boldog vagyok ma!"},
                "sad":      {"file": "hu_man_sad.wav",      "text": "Nagyon szomorú vagyok... Egyedül szeretnék lenni."},
                "surprised":{"file": "hu_man_surprised.wav","text": "Nem hiszem el! Micsoda hihetetlen meglepetés!"},
            },
        },
        "woman_soft": {
            "name": "Eszter",
            "emotions": {
                "angry":    {"file": "hu_woman_angry.wav",    "text": "Elég! Nem bírom tovább ezt a helyzetet!"},
                "happy":    {"file": "hu_woman_happy.wav",    "text": "Annyira boldog vagyok! Minden csodálatosan alakul!"},
                "sad":      {"file": "hu_woman_sad.wav",      "text": "Olyan szomorú vagyok ma... Egyedül kell lennem."},
                "surprised":{"file": "hu_woman_surprised.wav","text": "Istenem! Micsoda csodálatos meglepetés!"},
            },
        },
    },

    "ja": {
        "man_calm": {
            "name": "ひろし",
            "emotions": {
                "angry":    {"file": "ja_man_angry.wav",    "text": "もう限界だ。何度も同じことを言わせないで。"},
                "happy":    {"file": "ja_man_happy.wav",    "text": "今日は本当に気分がいい。すべてがうまくいっている。"},
                "sad":      {"file": "ja_man_sad.wav",      "text": "今日は少し落ち込んでいる。少し一人にしてほしい。"},
                "surprised":{"file": "ja_man_surprised.wav","text": "えっ、本当に？まったく予想していなかった。"},
            },
        },
        "woman_soft": {
            "name": "ゆき",
            "emotions": {
                "angry":    {"file": "ja_woman_angry.wav",    "text": "それはひどいよ。もう我慢できない。"},
                "happy":    {"file": "ja_woman_happy.wav",    "text": "とても嬉しい。今日は素晴らしい一日になりそう。"},
                "sad":      {"file": "ja_woman_sad.wav",      "text": "少し悲しい気持ち。今は静かに過ごしたい。"},
                "surprised":{"file": "ja_woman_surprised.wav","text": "わあ、びっくりした。本当にうれしい驚きだね。"},
            },
        },
    },

    "pl": {
        "man_confident": {
            "name": "Michał",
            "emotions": {
                "angry":    {"file": "pl_man_angry.wav",    "text": "Dosyć tego. Nie każ mi powtarzać tego po raz kolejny."},
                "happy":    {"file": "pl_man_happy.wav",    "text": "To wspaniały dzień. Jestem naprawdę szczęśliwy."},
                "sad":      {"file": "pl_man_sad.wav",      "text": "Dziś czuję się przygnębiony. Potrzebuję chwili spokoju."},
                "surprised":{"file": "pl_man_surprised.wav","text": "Naprawdę? Tego zupełnie się nie spodziewałem."},
            },
        },
        "woman_warm": {
            "name": "Agnieszka",
            "emotions": {
                "angry":    {"file": "pl_woman_angry.wav",    "text": "Mam dość tej sytuacji. To naprawdę frustrujące."},
                "happy":    {"file": "pl_woman_happy.wav",    "text": "Jestem taka szczęśliwa. Wszystko układa się znakomicie."},
                "sad":      {"file": "pl_woman_sad.wav",      "text": "Dziś jest mi bardzo smutno. Chcę pobyć chwilę sama."},
                "surprised":{"file": "pl_woman_surprised.wav","text": "O mój Boże, co za niespodzianka. Jestem zachwycona."},
            },
        },
    },
}


def language_to_code(language: str) -> Optional[str]:
    """Map OmniVoice language display name to a preset directory code."""
    return _LANG_TO_CODE.get((language or "").strip().lower())


def list_presets(language: str) -> List[Dict]:
    """Return preset metadata list for *language*, or [] if unsupported."""
    code = language_to_code(language)
    if not code or code not in PRESET_CATALOG:
        return []
    return [
        {
            "id": preset_id,
            "name": meta["name"],
            "emotions": sorted(meta["emotions"].keys()),
        }
        for preset_id, meta in PRESET_CATALOG[code].items()
    ]


def list_all_presets() -> Dict[str, List[Dict]]:
    """Return all presets grouped by language code."""
    return {
        code: [
            {"id": pid, "name": m["name"], "emotions": sorted(m["emotions"].keys())}
            for pid, m in presets.items()
        ]
        for code, presets in PRESET_CATALOG.items()
    }


def _find_preset_language_code(preset_id: str) -> Optional[str]:
    """Return the language code that owns *preset_id*, or None if not found.

    Used as a fallback when language is 'Auto' or otherwise unresolvable.
    If the same preset_id exists in multiple languages (unlikely but possible)
    the first match in catalog iteration order is returned.
    """
    for code, presets in PRESET_CATALOG.items():
        if preset_id in presets:
            return code
    return None


def resolve_preset(
    language: str,
    preset_id: str,
    emotion: str,
) -> Tuple[str, str]:
    """Return (audio_path, ref_text) for the given preset.

    Raises HTTPException 400/404 on invalid inputs or missing files.
    When *language* is 'Auto' or unknown the preset catalog is searched by
    preset_id so the caller does not have to pass an explicit language.
    """
    code = language_to_code(language)
    if not code:
        code = _find_preset_language_code(preset_id)
    if not code:
        raise HTTPException(
            status_code=400,
            detail=f"No voice presets available for language '{language}'.",
        )

    lang_presets = PRESET_CATALOG.get(code, {})
    if preset_id not in lang_presets:
        available = ", ".join(lang_presets.keys()) or "(none)"
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{preset_id}' for language '{language}'. "
                   f"Available: {available}",
        )

    emotions = lang_presets[preset_id]["emotions"]
    if emotion not in emotions:
        available = ", ".join(sorted(emotions.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown emotion '{emotion}' for preset '{preset_id}'. "
                   f"Available: {available}",
        )

    em = emotions[emotion]
    audio_path = str(PRESETS_DIR / code / preset_id / em["file"])

    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=404,
            detail=f"Preset audio file not found on server: {audio_path}",
        )

    return audio_path, em["text"]


def build_transcript_map() -> Dict[str, str]:
    """Flatten the catalog into {relative-posix-wav-path: ref_text}.

    Keys match the path of each preset WAV relative to the presets root
    (e.g. ``en/man_voice_deep/en_man_happy.wav``), so callers holding an
    opaque ``preset:<rel>`` id can look up the transcript directly.
    """
    transcripts: Dict[str, str] = {}
    for code, presets in PRESET_CATALOG.items():
        for preset_id, meta in presets.items():
            for em in meta["emotions"].values():
                rel = f"{code}/{preset_id}/{em['file']}"
                transcripts[rel] = em["text"]
    return transcripts
