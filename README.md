# HELOC-Risk-Prediction

HELOC stands for “Home Equity Line of Credit”, it is a loan set up as a line of credit for some maximum draw, rather than for a fixed dollar amount. For example, using a standard mortgage you might borrow $200,000, which would be paid out in its entirety at closing. Using a HELOC instead, you receive the lender’s promise to advance you up to $200,000, in an amount and at a time of your choosing. 

HELOCs are convenient for funding intermittent needs, such as paying off credit cards, making home improvements, or paying college tuition. Because the upfront costs are relatively low, many people choose HELOC as a source for funds. The challenge faced by financial institutions is how to identify the clients that are not able to repay the loan , which is critical for institutions to minimize the risk.  To overcome this challenge, an increasingly popular approach is machine learning-based methods, which are more time-efficient and cost-effective than manual annotation.

However, since financial institutions are required to provide explanations to clients about their rating, I built and compared several models, then chose the one with high prediction performance and interpretability using real- world financial datasets provided by FICO. Also, for users’ convenience, I created an interactive interface using Streamlit which also allows users to give clear explanations for the rating. 

The dataset contains HELOC applications made by real homeowners provided by FICO, in which RiskPerformance is the target binary variable. The label “Good” indicates that clients made payments without being more than 90 days overdue, whereas “Bad” indicates that they made payments 90 days past due or worse at least once over a period of 24 months since the credit account was opened. All the other predictor variables are quantitative or categorical variables representing a specific kind of trade feature of the homeowners. 

