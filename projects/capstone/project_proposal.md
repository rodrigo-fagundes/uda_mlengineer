# Machine Learning Engineer Nanodegree
## Capstone Proposal
Rodrigo Moreira Fagundes
December 14th, 2018

## Proposal

### Domain Background

In such a large developing country as Brazil, with so many different economic and cultural setups, tackling modern slavery efficiently is a huge challenge. According the Observatório Digital do Trabalho Escravo (https://observatorioescravo.mpt.mp.br/), from 2003 to 2018, 44,229 people were rescued from degrading working conditions in 3,318 inspections (13.33 rescues per inspaction). 2,006 successful diligences took place in 766 of the 5,570 brazilian municipalities (a coverage of 13.75%). If we add the 1,847 inspections with no rescues, the coverage raises to a 37.9%, with 2,112 locations, but it reveals a success rate of 52.06%. Since 2015, the number of diligences have benn falling, returning to the yearly inspections frequency seen in 2003-2007.

Some regions are not accessible, taking too long to mobilize an inspection - allowing perpetrators to move their operations or just hide their illegal aspects during the audit. Traditionally, the inspections are mobilized based on a denounce and evidences that support it. The most vulnerable people, though, lack the opportunity to reach government agencies.

In opposition, urban centers with high population density have a large number of enterprises to be verified, rendering a traditional coverage goal unrealistic - both due to cost and manpower. Modern slavery practices in urban centers tend to be disguised as just poor labor standards or practices, facing long disputes before estabilishing the perpetration.

### Problem Statement

Given the scenarios, government agencies have to craft a way to concentrate its resources in targets that reach more vulnerable people. One way to create such a prioritization can be trying to identify, based on municipalities profiles and the record of previous inspections, the locations that would most likely to result in more rescues per diligence.

### Datasets and Inputs

For this study, we're using three datasets. The first is a collection of information on municipality's census, collected by IBGE (Instituto Nacional de Geografia e Estatística), available to the public. The comprehensive dataset is a suitable source for identifying profiles and similarities between locations.

The second and third datasets will be the disidentificated registers of operations and inspections. They contain information on the municipalities where inspections took place, how many people were rescued from degrading work conditions, their origin and where they claimed to reside at the moment. In the current study, they'll be used as a base for risk rating, which in turn will become the label for classification.

### Solution Statement

One solution to the problem can be resource optimization by defining high priorities munuicipalities based on statistical inference. By using municipalities similarities and previous diligences data, it's reasonable to focus on locations that are most likely to result in a more effective action, rescuing more people in a single inspection, for instance. By prioritizing municipalities according to the distribution of rescues per inspection, the model can be repeated, hopefully with decreasing numbers of perpetrations.

### Benchmark Model

The benchmark model is the actual rating of rescues per inspection exclusively using the traditional resource placing strategies.

### Evaluation Metrics

To measure the success of the solution, the municipalities that have no previous records flagged by the solution should be split in control and test subjects. The testing group should be subject to a task force and the result should be compared to the control (disclosed at the end of the evaluation, to avoid bias). A number of false positives should be expected, and compared to the metrics revealed in the modeling phase.

If the rescues per inspections in the testing group is higher than the control group (error margin considered), the model is proven effective. The result should also be compared to the overall rating and to the previous records - it can reveal a migration of modern slavery practices.

### Project Design
_(approx. 1 page)_

First, we'll estabilish three menace rating (LOW, MEDIUM, HIGH), using the register of previous inspections using the distribution of rescues per inspection. The levels will be:

- LOW : the first quartile;
- MEDIUM : interquartile interval and
- HIGH : last quartile.

After that, we'll label the municipalities that actually had inspections according to these ratings and the average of the rescues per inspection in them. This rating will be added to the census data and an algorithm will be run to reduce the dimensionality, based on the information gain.

With the dimensionality reduced, we'll run a classification algotrithm to build the model for future prediction (on municipalities that have no inspection record).

Finally, the model will be applied for labeling municipalities with no inspection record. From that result, the locations with a rating of HIGH would join the previous HIGH labeled ones as flagged.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
