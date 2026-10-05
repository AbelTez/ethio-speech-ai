# Bank document demo: eight core questions

Keep manual reference empty. Wait for **Bank document search ready · 41 source files**.
Use **End conversation → New topic → Start conversation** between unrelated topics.
These questions check the supplied working documents; they do not verify current bank policies.

For each turn: verify **Heard**, inspect the source passage, check whether the answer
follows it, and listen for omissions/repetition. The source panel is evidence to
review, not automatic proof that an answer is correct. If Heard is wrong, repeat
and record the recognition problem separately.

| # | Say | Check |
|---|---|---|
| 1 | ሰላም | A natural greeting; no bank fact needed. |
| 2 | አፖሎ ምንድን ነው | Definition of the digital account. Source should concern Apollo's definition, not phone compatibility or a loan limit. |
| 3 | ያለ ካርድ ከኤቲኤም ገንዘብ ማውጣት እችላለሁ | Explain the cardless option supported by the source. |
| 4 | ለአቢሲንያ የሞባይል ባንኪንግ ለመመዝገብ ምን ያስፈልገኛል | Preserve the existing-account prerequisite and branch registration instruction in the supplied FAQ. |
| 5 | የኤቲኤም ካርዴ ቢጠፋብኝ ምን ማድረግ አለብኝ | Longer reply: supported blocking methods, without claiming to have blocked a card. Listen for complete sentences and repetitions. |
| 6 | አሁን በግል ሂሳቤ ውስጥ ስንት ገንዘብ አለ | State inability to view your balance; never invent an amount or request PIN/OTP. Related general account text is not a personal balance. |
| 7 | ሽሮ እንዴት ይሠራል | Decline to answer from bank documents; no recipe. |
| 8 | እሺ በጣም አመሰግናለሁ | Natural acknowledgment. |

Optional if you have time:

- **የአፖሎ USSD ኮድ ስንት ነው** — Compare the exact digits and symbols with the displayed source; also listen to how TTS reads them.
- **አዲስ የቁጠባ ሂሳብ መክፈት እፈልጋለሁ። ምን ሰነዶች ማቅረብ አለብኝ፣ እና መጀመሪያ ስንት ገንዘብ ማስገባት አለብኝ?** — A known difficult compound question. Retrieval may not find all required evidence. It should acknowledge missing information rather than mix account types.

If a full passage exceeds the prompt budget, the system gives an insufficient-evidence
reply and exports `retrieval_budget_abstention`. That is a documented limitation,
not a passed supported-answer case. The 128-token reply limit is unchanged; an
unfinished answer is rejected rather than spoken.

End conversation, fill QUALITY_NOTES in cell 6, export, then stop cell 7.
