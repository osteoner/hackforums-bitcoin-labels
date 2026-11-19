# Data statement

## Source
Forum data come from the CrimeBB academic archive (HackForums), provided by the Cambridge Cybercrime Centre
under its Data Use Agreement. The agreement prohibits redistributing raw post text or user profiles, so this
repository contains none. Anyone who wants to rerun the extraction and screening scripts must obtain CrimeBB
access from the Cambridge Cybercrime Centre.

## What is released
- Bitcoin addresses, which are public on the blockchain, and on-chain statistics derived from public
  blockchain data.
- The cybercrime category from LLM screening and structured screening outputs (votes, confidence, category).

## What is not released
- Post text, thread titles, user names or user IDs.
- LLM justifications and raw LLM answers, because they can quote or paraphrase post text.

## Human subjects
The study is non-interventional. It uses archived, publicly visible forum posts only; no forum user was
contacted, no account was created and no transaction was made.

## Sensitive content
Some posts advertise illegal goods and services, including a small number of references to sexual
exploitation. Review was limited to reading archived plain text; hyperlinks, attachments and darkweb
domains were not opened. Addresses linked to such content are released only with an aggregate category
(`Abuse`).

## Intended use
Research on cryptocurrency-facilitated cybercrime, illicit address detection and blockchain analytics.
Labels come from forum context and automated screening and can contain errors. They must not be used as
the sole basis for action against any person or address.
