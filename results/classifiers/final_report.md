# Final scoring: 40 conversations

Judges: Rachel voice = Qwen3-8B LoRA (Tinker); partner = TF-IDF + logistic regression (6 partners).

| | Real Rachel line | Generated line | Difference (95% CI) |
|---|---|---|---|
| Mean P(Rachel's voice) | 0.56 | 0.38 | +0.18 (+0.12 to +0.24) |
| Lines judged Rachel (P > 0.5) | 26/40 | 10/40 | |
| Ross vs Monica classified correctly | 0.68 | 0.55 | +0.12 (-0.07 to +0.33) |
| Mean P(true partner), 6-way | 0.23 | 0.18 | |

Ross vs Monica classification (rows = true partner):

| | Real: →Ross | Real: →Monica | Generated: →Ross | Generated: →Monica |
|---|---|---|---|---|
| Ross | 14 | 6 | 13 | 7 |
| Monica | 7 | 13 | 11 | 9 |

## Per conversation

| id | partner | real P(R) | real → | gen P(R) | gen → | generated line |
|---|---|---|---|---|---|---|
| s03_e05_c14_u001 | Ross | 0.56 | Monica | 0.56 | Monica | [Other Person], let's not forget who's been playing the long game here. |
| s02_e07_c11_u001 | Ross | 0.41 | Ross | 0.32 | Ross | Well, everyone has their own preferences. If it's not something that sparks your interest,… |
| s01_e05_c12_u001 | Ross | 0.59 | Ross | 0.56 | Ross | Oh no, not the pink laundry again! Let me see. (laughs) Person E, you know I can't help it… |
| s01_e09_c10_u001 | Monica | 0.53 | Ross | 0.24 | Ross | Are you sure? I remember you saying you had them earlier. We can't cook the turkey if we c… |
| s03_e25_c05_u001 | Monica | 0.65 | Ross | 0.59 | Ross | [Laughs] No, not at all. We're just discussing some design ideas for her new shoes. |
| s04_e24_c19_u001 | Monica | 0.71 | Monica | 0.59 | Ross | Yes, now is good. I was just wondering, do you think you could help me out with something … |
| s05_e20_c08_u001 | Monica | 0.56 | Ross | 0.47 | Ross | Person C: Where to, though? I'd like to grab a coffee first if that's okay with you. |
| s05_e07_c09_u001 | Monica | 0.44 | Monica | 0.35 | Monica | I'm sorry, but I can't do that. The gala starts at 6:30 sharp, and I have important obliga… |
| s02_e02_c08_u001 | Monica | 0.71 | Monica | 0.29 | Ross | I need some time to process this. Let's talk about it later when I can think more clearly. |
| s04_e21_c28_u001 | Monica | 0.68 | Monica | 0.22 | Monica | Well, then I'll need to find a way to get there. I can't let Phoebe down. She's counting o… |
| s09_e20_c01_u000 | Monica | 0.44 | Monica | 0.38 | Ross | Well, I mean, it really depends on the individuals involved and their dynamic, doesn't it?… |
| s02_e08_c05_u001 | Monica | 0.32 | Ross | 0.61 | Monica | Well, then, what do you want? |
| s10_e17_c02_u000 | Ross | 0.53 | Ross | 0.32 | Monica | That's interesting. Maybe you should put those new moves to good use. Good luck at work! |
| s01_e24_c08_u001 | Monica | 0.53 | Ross | 0.68 | Monica | Person C gave me a beautiful skirt for my birthday. |
| s03_e15_c04_u001 | Ross | 0.76 | Ross | 0.44 | Ross | Not much, just had a long day at work. How about you? [Other Person]: Same here. I've been… |
| s02_e01_c07_u001 | Ross | 0.42 | Monica | 0.38 | Monica | Well then, I'll just go handle the situation. Thanks for letting me know. |
| s10_e09_c05_u000 | Ross | 0.83 | Ross | 0.62 | Ross | Person C: Wait, where are you going? Person A: Why don't you tell me, Person C? What's goi… |
| s05_e01_c08_u001 | Ross | 0.65 | Ross | 0.32 | Ross | No problem, I'll go back and get it for you. |
| s04_e24_c19_u001 | Monica | 0.73 | Monica | 0.32 | Ross | I see. Well, I'll make sure to be more careful with my actions moving forward. Thank you f… |
| s07_e16_c12_u001 | Ross | 0.68 | Monica | 0.59 | Ross | Person B, let's give them some space for now. We can talk about it later. |
| s09_e05_c12_u000 | Ross | 0.56 | Ross | 0.32 | Ross | Person E seems to be in need of attention. I'll be right here with you. Don't worry about … |
| s02_e05_c07_u001 | Ross | 0.62 | Ross | 0.32 | Monica | It was good, thanks for asking. I had a great time at the party. How about yours? |
| s01_e21_c02_u001 | Monica | 0.44 | Monica | 0.24 | Monica | You know what? Let's try this instead. Why don't we go out and buy those art supplies toge… |
| s02_e04_c11_u001 | Ross | 0.59 | Ross | 0.11 | Ross | I think I've got it now. Let's try something new. How about we surprise each other with sm… |
| s02_e02_c08_u001 | Monica | 0.59 | Monica | 0.35 | Ross | I need some time to process this. Can we talk about it later? |
| s08_e18_c04_u001 | Ross | 0.68 | Ross | 0.22 | Ross | That's a great idea! Let's make sure everything is perfect for Person D's wedding. I'll do… |
| s03_e12_c15_u001 | Ross | 0.38 | Ross | 0.53 | Ross | Really? And you're taking Person D with you? [Other Person]: Yeah, she seems nice. Person … |
| s04_e24_c19_u001 | Monica | 0.72 | Monica | 0.41 | Monica | [Other Person], what should I do? [Other Person]: Just uh, do the right thing. Person A: W… |
| s09_e21_c02_u000 | Monica | 0.58 | Monica | 0.50 | Ross | (looking at the object) I'm not sure. Can you tell me more about it? |
| s05_e22_c04_u001 | Monica | 0.38 | Ross | 0.24 | Monica | Really? Can I see it up close? [Other Person]: I'm not sure that's a good idea. Person A: … |
| s10_e15_c01_u000 | Ross | 0.68 | Ross | 0.12 | Ross | That sounds exciting! Shall we celebrate with a glass of champagne instead? I've heard the… |
| s09_e07_c13_u000 | Ross | 0.13 | Ross | 0.29 | Ross | Well, I guess we're in sync today. Let's keep the positive energy going! |
| s10_e04_c02_u000 | Ross | 0.45 | Ross | 0.20 | Monica | Person D isn't here with us in this conversation. Let's focus on the present situation. Ho… |
| s04_e04_c01_u001 | Monica | 0.71 | Monica | 0.50 | Ross | Oh, I'm so sorry. I must have missed seeing that it was empty. I'll take it out and recycl… |
| s10_e15_c12_u000 | Ross | 0.65 | Monica | 0.22 | Monica | I'll have to think about it. Thanks for the advice. |
| s05_e05_c06_u001 | Ross | 0.48 | Monica | 0.36 | Ross | Sure thing, go ahead. I'm here to listen. |
| s01_e01_c01_u001 | Monica | 0.50 | Monica | 0.59 | Monica | I know, I know. But I couldn't help it. I felt like I needed to tell someone, and you're t… |
| s02_e01_c07_u001 | Ross | 0.38 | Monica | 0.29 | Monica | Who is "he" that you're referring to? I'd like to make sure I have all the information bef… |
| s01_e21_c02_u001 | Monica | 0.65 | Ross | 0.22 | Monica | You know what? Let's not compare, let's create instead. I'll help you get started with you… |
| s02_e08_c05_u001 | Monica | 0.45 | Monica | 0.29 | Ross | Hey there! How's your day going? Is there something specific you'd like to talk about? I'm… |
