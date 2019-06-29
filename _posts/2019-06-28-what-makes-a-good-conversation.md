---
layout: post
title: "What makes a good conversation?"
subtitle: "How controllable attributes affect human judgments"
hide: yes
---

<!--excerpt.start-->
*This blog post is about the [NAACL 2019](https://naacl2019.org/) paper __What makes a good conversation? How controllable attributes affect human judgments__ by Abigail See, Stephen Roller, Douwe Kiela and Jason Weston.*
[[paper](https://www.aclweb.org/anthology/N19-1170)] [[code/demo](https://parl.ai/projects/controllable_dialogue)] [[slides](https://cs.stanford.edu/people/abisee/naacl2019slides.pdf)]
<!--excerpt.end-->

---

### The Natural Language Generation task spectrum

When I think about Natural Language Generation (NLG) tasks, I imagine them on the following spectrum:[^sasha]

[^sasha]: Sasha Rush showed a similar diagram during his talk at the NeuralGen 2019 workshop. See "Open Questions" slide [here](http://nlp.seas.harvard.edu/slides/Pre-training%20for%20Generation.pdf).

{% include image.html
  img="controllable_dialogue/NLG_spectrum.png"
  alt="TODO"
  caption=""
  width=600
%}

On the left are tasks like Machine Translation (MT), which are **less open-ended** (i.e. there is a relatively narrow range of correct outputs given the input).
Given the close correspondence between input and output, these tasks can be accomplished mostly (but not entirely) by decisions at the word/phrase level.
On the right are tasks like Story Generation and Chitchat Dialogue, which are **more open-ended** (i.e. there is a huge range of appropriate outputs given the input).
For these tasks, the ability to make high-level decisions (e.g. 'what should happen next in the story?' or 'should we change the subject of discussion?') is central to the task.

While **neural Language Model (LM)** based approaches have so far been successful for tasks on the left, they have well-documented difficulties with tasks on the right, such as repetitious and generic output (under certain decoding algorithms).
More broadly, neural LMs seem to struggle to learn to make the necessary high-level decisions.

For this reason, **control** -- that is, the ability to specify desired attributes of the text at test time -- is an attractive idea for open-ended neural NLG.
For example, if we can control the repetitiveness or genericness of the text, we can fix the aforementioned errors.
Furthermore, if we can control certain high-level attributes of the text (e.g. whether to change the subject, or whether to ask a question), then perhaps we can make some high-level decisions _for_ the neural LM.

It is well-established that all NLG **evaluation** is difficult -- for example, the MT and summarization communities continue to use the BLEU and ROUGE automatic metrics despite their well-documented problems.
But for _open-ended_ NLG, evaluation is even more difficult.
In the absence of a useful automatic metric to capture overall quality, we rely on human evaluation.
But even that is complex -- when evaluating dialogue, should we evaluate single turns or multiple turns?
Should evaluators take part in conversations interactively or not?
What questions should be asked, and how should they be phrased?

### Three research questions

In this work, we use chitchat dialogue as a setting to better understand the issues raised above.
In particular, we **control multiple attributes of generated text** and **human-evaluate multiple aspects of conversational quality**, in order to answer **three main research questions**:

**[Research Question 1](#research-question-1-how-effectively-can-we-control-the-attributes): How effectively can we control the attributes?**
<br>
**Quick answer**: Pretty well! But some control methods only work for some attributes.

**[Research Question 2](#research-question-2-how-do-the-controllable-attributes-affect-conversational-quality-aspects): How do the controllable attributes affect conversational quality aspects?**
<br>
**Quick answer**: Strongly -- we get improvements by controlling repetition, question-asking, and specificity vs genericness.

**[Research Question 3](#research-question-3-can-we-use-control-to-make-a-better-chatbot-overall): Can we use control to make a better chatbot overall?**
<br>
**Quick answer**: Yes! Though the answer can depend on the definition of 'better overall'.


### The PersonaChat task

We use [PersonaChat](https://arxiv.org/pdf/1801.07243.pdf), a chitchat dataset containing conversations between two participants, each of which has a 'persona'.
Our task is to build a chatbot that can converse with a human in this setting.

{% include image.html
  img="controllable_dialogue/personachat.png"
  alt="TODO"
  caption="In the PersonaChat task, both participants are supplied with a persona and instructed to get to know each other."
  width=600
%}

The PersonaChat task was the focus of the [NeurIPS 2018 ConvAI2 Competition](http://convai.io/).
Most successful teams built neural sequence generation systems (see the [competition report](https://arxiv.org/pdf/1902.00098.pdf)).
In particular the winning team, _Lost in Conversation_, used a finetuned version of OpenAI's [GPT](https://openai.com/blog/language-unsupervised/) language model, which is pretrained on a very large amount of text (985 million words).

We use a simple baseline -- a standard LSTM-based sequence-to-sequence architecture with attention.
On each turn, the bot's persona is concatenated with the dialogue history to form the input sequence, and the output is generated using beam search.
<!-- comment on why beam search -->
We pretrain this model on 2.5 million Twitter message/response pairs, then finetune it on PersonaChat.

### Four controllable attributes of text

{% include image.html
  img="controllable_dialogue/controllable_attributes.png"
  alt="TODO"
  caption="We control four attributes of the output text."
  width=600
%}

Neural LMs often produce repetitive, generic or irrelevant text, especially when decoding using beam search.
Motivated by this, we control the **repetitiveness**, **specificity** and **response-relatedness** of the output text.
These attributes are defined simply: repetitiveness as n-gram overlap, specificity as word rareness, and response-relatedness as the embedding similarity of the bot's response to the human's last utterance.

Lastly, we also control the rate at which the bot asks **questions** (here we regard an utterance to contain a question if and only if it contains '?').
Question-asking is an essential component of chitchat, but one that must be balanced carefully.
By controlling question-asking, we can find and understand the right balance.

<!-- do we need to better motivate why these? -->

<!-- should we explain that the control is dialogue level not utterance level? -->

### Aspects of conversational quality

In our evaluations, we ask Amazon Mechanical Turk crowdworkers ('Turkers') to chat with our bots for six turns before asking them to rate several different aspects of the conversation (most are on a scale from 1 to 4).

{% include image.html
  img="controllable_dialogue/quality_aspects_low.png"
  alt="TODO"
  caption="We collect human evaluations for six lower-level aspects of conversational quality."
  width=600
%}

Some of the aspects -- such as **avoiding repetition**, **making sense**, and **fluency** -- are designed to capture certain basic error classes (like repeating, saying nonsensical things, or disjointed language).
The others -- **interestingness**, **listening**, and **inquisitiveness** -- encompass other important elements of conversation, each of which must be balanced.

{% include image.html
  img="controllable_dialogue/quality_aspects_high.png"
  alt="TODO"
  caption="We also collect human evaluations for two definitions of overall quality - humanness and engagingness."
  width=600
%}

Lastly, we ask the Turker to rate the bot with respect to two different notions of overall quality.
To measure **humanness**, we ask the Turker whether they think they spoke to a bot or a human (i.e. a Turing test question).
To measure **engagingness**, we ask the Turker how much they enjoyed the conversation.

Many dialogue studies use either engagingness or humanness as a single stand-alone quality metric.
<!-- need to cite? -->
In particular, in the ConvAI2 competition, only engagingness was used for human evaluation.
Given that we use the exact same wording of the engagingness question, our evaluation is a _superset_ of ConvAI2's.


<!-- add more motivation for why measure these things -->

### Control methods

In the recent neural sequence generation literature, there are many proposed methods to generate text with some desired attribute.
<!-- (see paper for citations) -->
However, due to NLG evaluation difficulties, it's not clear which of these control methods is most effective.
Furthermore, many of these methods are attribute-specific -- meaning they're designed to control just one particular attribute of the text (e.g. specificity), rather than any general attribute.

In this work, we use two simple existing methods for general-purpose control, and use them to control all four text attributes.
Aside from helping us to build a better chatbot, this also allows us to better understand the relative effectiveness of the control methods themselves.

<!-- (explain why we went with one control setting per dialogue not per utterance.) -->

#### Control method 1: Conditional Training (CT)

A standard sequence-to-sequence model learns $$P(y \vert x)$$, the conditional probability of the output text $$y$$ given the input text $$x$$.

A Conditional Training model learns $$P(y\vert x,z)$$, the conditional probability of the output text $$y$$ given the input text $$x$$ _and_ a control variable $$z$$, which specifies the desired output attribute.
For example, to control specificity, we might set $$z$$ to HIGH or LOW to get a very specific or a very generic response to _What's your favorite hobby?_

{% include image.html
  img="controllable_dialogue/CT.gif"
  alt="TODO"
  caption="Controlling specificity with Conditional Training"
  width=700
%}

The CT model is trained to predict $$y$$ given $$x$$ and $$z$$ (where $$z$$ is provided via automatic annotation).
Then at test time, $$z$$ can be chosen by us.

Several researchers have proposed versions of this method ([Kikuchi et al 2016](https://aclweb.org/anthology/D16-1140), [Peng et al 2018](https://aclweb.org/anthology/W18-1505), [Fan et al 2018](https://aclweb.org/anthology/W18-2706)), using various methods to incorporate $$z$$ into the model.
We represent $$z$$ with a learned embedding, and find that concatenating $$z$$ to each decoder input is most effective.
We can even concatenate _multiple_ control embeddings $$z_1, z_2, ..., z_n$$ and learn $$P(y \vert x, z_1, z_2, ... z_n )$$ if we wish to simultaneously control several attributes.


#### Control method 2: Weighted Decoding (WD)

Weighted Decoding ([Ghazvininejad et al 2017](https://aclweb.org/anthology/P17-4008), [Baheti et al 2018](https://aclweb.org/anthology/D18-1431)) is a simple technique, applied during decoding, to increase or decrease the probability of words with certain _features_.

For example, to control specificity with Weighted Decoding, we use the rareness of a word as a feature.
On each step of the decoder, we update the probability of each word in the vocabulary, in proportion to its rareness.
The size of the update is controlled by a weight parameter, which we choose -- allowing us to encourage more specific or more generic output.
In the example below, we increase the probability of rarer words, thus choosing _I like watching sunrises_ rather than _I like watching movies_.

{% include image.html
  img="controllable_dialogue/WD.gif"
  alt="TODO"
  caption="Controlling specificity with Weighted Decoding"
  width=700
%}

This method requires no special training and can be applied to modify any decoding algorithm (beam search, greedy search, top-k sampling, etc).
Weighted Decoding can be used to control multiple attributes at once, and it can be applied alongside Conditional Training.

### Research Question 1: How effectively can we control the attributes?

We find that **Weighted Decoding** is effective to control attributes that can be easily defined at the word-level, like <font color="#0f9d58">repetition, specificity</font>, and <font color="#0f9d58">response-relatedness</font> (shown below).
However, the method yields degenerate output when the feature weight is too high -- for example, devolving into a long list of related words (_drinks, espresso, latte, tea_).

{% include image.html
  img="controllable_dialogue/controlling_response_rel.png"
  alt="TODO"
  caption="Controlling response-relatedness using Weighted Decoding (WD). By increasing response-relatedness, we obtain a more on-topic response (<i>I do, usually at starbucks</i>)."
  width=600
%}

Because Weighted Decoding controls attributes using word-level features, it cannot control attributes such as <font color="#db4437">question-asking</font>, which are more naturally defined at the sentence-level.
<!-- more details on this? -->

We find that **Conditional Training** is effective to control simple attributes of the output text, such as <font color="#0f9d58">specificity</font> and <font color="#0f9d58">question-asking</font>.
In particular, it usually produces output that is well-formed and has the desired attribute -- this makes it less risky than Weighted Decoding (see below for example).

{% include image.html
  img="controllable_dialogue/controlling_specificity.png"
  alt="TODO"
  caption="Controlling specificity using Weighted Decoding (WD) and Conditional Training (CT). By increasing specificity, we obtain more interesting, personalized responses."
  width=600
%}

However, we find Conditional Training is less effective at learning to control _relationships_ between the input and output, such as <font color="#db4437">response-relatedness</font>.
In addition, Conditional Training can't control attributes without sufficient training data -- meaning it is ineffective to control <font color="#db4437">repetition</font>, because our training data does not contain the kind of severely repetitive output we wish to prevent.

Overall, though the control methods didn't work for every attribute, we find that each of our four attributes can be satisfactorily controlled by at least one of the two methods.

### Research Question 2: How do the controllable attributes affect conversational quality aspects?

We find that __reducing repetition__ gives large boosts to <font color="#0f9d58">all human evaluation scores</font>.
This is not surprising, as our beam search baseline model repeats itself a lot (especially across utterances), creating a very frustrating user experience.
However, this does demonstrate the importance of multi-turn evaluation (vs single response generation), which is needed to detect across-utterance repetition.

By __increasing specificity__ to around human levels, we obtain improvements to <font color="#0f9d58">interestingness, listening</font> and <font color="#0f9d58">engagingness</font>.
However, finding the right balance is difficult -- increasing specificity too much leads to lower <font color="#db4437">making sense</font> and <font color="#db4437">fluency</font> scores.

We also find that by __increasing question-asking__ rate to 65.7%, we achieve better <font color="#0f9d58">inquisitiveness, interestingness</font> and <font color="#0f9d58">engagingness</font>.
Interestingly, this rate is higher than both the baseline (50%) and humans (28.8%) -- implying that, in chitchat settings such as these, more question-asking is often received well.

Lastly, we were unable to obtain an improvement in any of our evaluation categories by controlling __response-relatedness__.
Though we hoped that increasing response-relatedness would create a chatbot that appears more attentive, friendly and interested in the user, Turkers did not rate the 'more responsive' bots well.
In particular, these bots received lower scores for <font color="#db4437">fluency</font> and <font color="#db4437">making sense</font>, and consequently lower overall scores for <font color="#db4437">humanness</font> and <font color="#db4437">engagingness</font> too.
As with specificity, attempting higher response-relatedness is a risky strategy, as it increases the chance of the bot saying something that sounds unnatural or nonsensical.

<!-- add example conversations or talk about where they can be found, for this section -->

### Research Question 3: Can we use control to make a better chatbot overall?

The first answer is __yes__!
By controlling repetition, specificity and question-asking, we achieve
__near-human engagingness__ (i.e. enjoyability) ratings.

{% include image.html
  img="controllable_dialogue/engagingness.png"
  alt="TODO"
  caption="Engagingness (i.e. enjoyability) ratings for humans and selected models."
  width=600
%}

In particular, our raw engagingness score matches that of the ConvAI2 competition winner's GPT-based model.[^convai2]
This is especially notable because our model is much smaller (a 2-layer LSTM-based model vs 12-layer Transformer-based model), and is trained on 12 times less data.
<!-- TODO: change this to number of params. -->

[^convai2]: Though we used the exact same wording as ConvAI2 for our Engagingness question, the comparison of raw scores should be considered as a rough indication of a similar overall quality, _not_ an exact comparison.

However, on the __humanness__ (i.e. Turing test) metric, all our models are __nowhere near human-level__!

{% include image.html
  img="controllable_dialogue/humanness.png"
  alt="TODO"
  caption="Humanness (i.e. Turing test) ratings for humans and selected models."
  width=600
%}

We've observed that __our bots are (almost) as engaging as humans, but they're clearly non-human__.
What does this mean?

Firstly, our results demonstrate that __engagingness is not the same as humanness__.
While both metrics are frequently used alone for evaluation, our results show the importance of measuring more than one.

Secondly, we suspect that on this task, the __human 'engagingness' performance may be artificially low__.
This is because Turkers chatting for money, using artificial personas, are less engaging conversationalists than people who are genuinely chatting for fun.
This may explain why the human-level engagingness scores are easy to match.

<!-- Show example of engaging but not human-like bot? Show example of human-like but not engaging turker? -->

### Conclusions

* **Control is a good idea** for your neural sequence generation dialogue system. Using simple control, we matched the performance of a GPT-based contest winner. We expect these techniques would yield even better results when applied to a highly pretrained language model like GPT.
* We investigated **two general-purpose control methods** which have complementary strengths and weaknesses. If you want to control a fairly simple attribute of the output text, and you have sufficient training examples of the attribute, then **Conditional Training** is probably a good idea. If you don't have the training data, or the attribute is harder to learn, then **Weighted Decoding** may be more effective -- though you need to be careful as the method can produce degenerate output.
* **Multi-turn phenomena** (such as repetition across utterances, and question-asking frequency) are important to conversations – so we need **multi-turn eval** to detect them.
* **Engagingness is not the same as humanness**, so think carefully about which to use as an overall quality metric.
* **Paid Turkers are not very engaging conversationalists**, and perhaps aren't even good judges of whether a conversation is engaging.
Though it raises other evaluation challenges, humans chatting for fun may be a better source of genuine judgments.
* Whether you're a human or a bot: **Don't repeat yourself. Don't be boring. Ask more questions.**

### Outlook

This project involved a lot of manual tuning of control parameters, as we attempted to find the best combination of settings for the four attributes.
This was a long and laborious process, requiring not only many expensive hours of Turker evaluation time, but also many hours of our own evaluation time as we chatted to the bots.

I'm reminded of [QWOP](http://www.foddy.net/Athletics.html) -- a simple game in which you press four buttons (Q, W, O and P) to control the individual muscles in a runner's legs.
Though the aim of the game is to run as far as possible, the entertainment comes from the absurd difficulty of the task.

{% include image.html
  img="controllable_dialogue/qwop.gif"
  alt="TODO"
  caption="QWOP is a game in which you attempt to run by pressing four buttons that each control a different part of the runner's legs."
  width=400
%}

Manually controlling four low-level text attributes is _not_ the most principled, nor the most scalable way to build a good conversational dialogue system -- just as manually controlling the four parts of the runner's legs is not the most principled way to run a marathon.
<!-- Though identifying better ways to control text (e.g. tuning the control settings automatically) is a great area for future work. -->
However, for the neural sequence generation systems we are using today, this kind of control can be useful and effective -- getting us a little further down the track, if not all the way to the finish line.



<!-- The attributes we controlled were all easy to define and measure automatically. there's a reason we can't do harder stuff. yejin choi said that maybe the most important thing right now is to learn to automatically measure harder things like coherence.
Beam search vs sampling.
what about dialogue manager? -->



---

*For further details on this work, check out the [paper](https://www.aclweb.org/anthology/N19-1170).*

*If you'd like to chat to the bots yourself, follow the instructions [here](https://parl.ai/projects/controllable_dialogue) -- it only takes a few minutes to set up!*

---

Footnotes
