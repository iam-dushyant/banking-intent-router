# Introduction
This project considers a banking chat use case, where thousands of messages come in, for example, "my card hasn't arrived", "why was I charged twice", etc. Each one needs to reach the right team fast, and some of the complex queries would need to be routed to a human advisor.

# The pipeline
When the customer sends a message, a sentence encoder is used to classify intent. This is, then, forwarded to a router to classify whether to route this message to a team or ask further questions to the customer.