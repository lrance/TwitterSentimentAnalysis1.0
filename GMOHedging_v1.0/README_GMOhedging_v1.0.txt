==== Introduction ====

This README v1.0 (June, 2012) for the Cornell GMOhedging v1.0 comes
from the URL
https://confluence.cornell.edu/display/llresearch/HedgingFramingGMOs .

The data is potentially relevant to research on (a) framing,
especially with respect to the debate over the use of genetically modified
organisms (GMOs) and/or the differences between "professional-science"
and "pop-science" discourse, and (b) hedging.

==== Citation info: ====

This data was first used in:

@InProceedings{Choi+al:2012, 
  author =   {Eunsol Choi and Chenhao Tan and Lillian Lee and Cristian Danescu-Niculescu-Mizil and Jennifer Spindel}, 
  title =    {Hedge detection as a lens on framing in the {GMO} debates: A position paper}, 
  booktitle = {Proceedings of the Workshop on Extra-Propositional Aspects of Meaning in Computational Linguistics}, 
  year =     {2012} 
} 

==== Zip file contents ====

This README (README.txt), plus the following files:

    gmostyle.pdf 154K   [paper describing our work]
    lexis 4.8M ["pop science"]
    wos 893K ["professional science"]
    pro_GMO 4.7M
    processed_pro_GMO 500K
    anti_GMO 4.2M
    processed_anti_GMO 514K
    corpus_table 35K
    sample_annotation 39K

====  lexis  ====

Format: 

928 raw documents contained within a single text file, delimited by
lines of the form "Document Number: n", where n ranges from 1 to 928
inclusive.  Line breaks from the original LexisNexis source are
indicated by ascii code: 13 (these may appear as "\r" or "^M").  Other
control characters, non-standard characters, and document headers and
footers may have also been retained.


Derivation: 

These 928 documents were collected from the LexisNexis database
using the search keywords "genetically modified foods" or
"transgenic crops" with search limited to US newspapers. Then we
eliminated articles that do not contain at least two occurrences of
the following keywords: GMO, GM, GE, genetically modified, genetic
modification, modified, modification, genetic engineering, engineered,
bioengineered, franken, transgenic, spliced, G.M.O., tweaked,
manipulated, engineering, pharming, aquaculture. We also eliminated
articles containing over 2000 words. 

==== wos ====

Format: 

648 raw abstracts (considered as documents) contained within a single text
file, delimited by lines of the form "Document Number: n", where n
ranges from 1 to 648 inclusive.  Line breaks from the original source
are preserved (not indicated by control characters).

Derivation: 

From Thomson Reuter's Web of Science (WOS), a database of
scientific journal and conference articles, 648 scientific paper
abstracts were collected using "transgenic foods" as a search keyword.
We discarded results containing either of the two off-topic filtering terms
"mice" or "rats". After this, we manually removed off-topic texts from
the collection. 


==== {anti,pro}_GMO ====

Format:  

Raw (but plaintext, no html markup) documents contained within a
single text file, delimited by lines of the form "Document Number: n"
where n is a random number between 1 and 9999 inclusive.  Each line
represents a paragraph, and each such line within a document is
separated from the next by a blank line.  (The "Document Number: n"
delimiter lines are not separated by a blank line from the first line
of the document).

Derivation: 

We used our (in particular, Jennifer Spindel's) domain expertise to compile a list of 20 anti-GMO and 20 pro-GMO
organization websites. After the initial collection of data from
these websites, near-duplicates and irrelevant articles were filtered
through clustering, keyword searches and distance between word vectors
at the document level. 'anti_GMO', a collection of articles
representing the opponents of GMO, contains 762 articles and
'pro_GMO', a collection of articles representing the proponents of
GMO, contains 671 articles. The mapping from
document number to its position (pro or anti) and source is provided in the
'corpus_table' file.

==== processed_*  files ==== 

Format: 

Same as in the anti_GMO and pro_GMO files, except that:

1. Each contains only  404 "pro" and 404 "anti" documents, to represent a balanced corpus

2. Each retained "document" consists of only the first 200 words after 
excluding the first 50 words of documents containing over 280 words. 
This was done to avoid irrelevant sections such as 
"Educators have permission to reprint articles for classroom use; 
other users, please contacteditor@actionbioscience.org for reprint permission. 
See reprint policy".

==== corpus_table ====

This file provides a mapping between document number and its source. 
Each line in corpus_table file has three columns, 
the first column representing document number, the second representing its
position ("pro" or "anti"), and the third giving its source.
Examples:

    3171     pro    biofortified
    3174     anti   center_for_food_safety

The following provides the urls of the website for each source name in the third
column (e.g., "biofortified" or "center_for_food_safety"). (These URLs might not work anymore, as the data was collected in summer
2011).

abbreviation	url
---------------------------------------------------------------------------
green_peace	http://www.greenpeace.org
natural_news	http://www.naturalnews.com/list_features_GMOs.html
say_no_to_gmo	http://www.saynotogmos.org/scientists_speak
gmo_watch	http://www.gmwatch.eu
soil_assoc	http://www.soilassociation.org/Whyorganic/GM/News
environment_common	http://environmentalcommons.org/
nano_transform	http://nanotransformation.com
organic_consumers	http://www.organicconsumers.org
responsible_technology	http://www.responsibletechonology.org
center_for_food_safety	http://www.centerforfoodsafety
sierra club	http://www.sierraclub.org
gmfree_scotland	http://gmfreescotland.blogspot.com/
non_gmo_project	http://www.nongmoproject.org/
psrast	http://psrast.org/
gmo_awareness	http://gmo-awareness.com/
gmo_journal	http://www.gmo-journal.com/
action_bioscience	http://www.actionbioscience.org/biotech/pusztai.html
harvest_of_fear_n	http://www.pbs.org/wgbh/harvest/
gmo_danger	http://userwww.sfsu.edu/~rone/GEessays/gedanger.htm
biofortified	http://www.biofortified.org/page/
agbioworld	http://www.agbioworld.org/newsletter_wm/
better_foods	http://www.betterfoods.org
biotechnow	http://www.biotech-now.org/food-and-agriculture
whybiotech	http://www.whybiotech.com
gmo_africa	http://www.gmoafrica.org
growers	http://www.growersforwheatbiotechnology.org/html/gwb_news.cfm
isaaa	http://www.isaaa.org/
golden_rice	http://www.goldenrice.org/
soy_connection	http://www.soyconnection.com/soybean_oil/benefits_of_biotechnology.php
biotechnology	http://www.bio.org/category/41
ncbe	http://www.ncbe.reading.ac.uk/NCBE/GMFOOD/menu.html
harvest_of_fear	http://www.pbs.org/wgbh/harvest/
monsanto	http://www.monsanto.com/newsviews/Pages/biotech-safety-gmo-advantages.aspx
bt	http://www.bt.ucsd.edu/gmo.html
who	http://www.who.int/foodsafety/publications/biotech/20questions/en/



====  sample_annotation  ====

This file contains 200 hand-annotated randomly-sampled sentences, half
from wos and half from lexis dataset.  It is delimited by lines of the
form "Sentence Number: n (1|-1|?) (1|-1|?) (LEXIS|WOS) ",  where n is
a random number between 1 and 200 inclusive. The second column
indicates the first annotator's opinion and the third column indicates the
second annotator's opinion, according to the following label scheme:
        1 = sentence is certain
	-1 = sentence is uncertain (contains hedging)
 	? =  not a proper sentence.  
The fourth column indicates the sentence's  source, either lexis or
wos.  (The annotators were not privy to this source information.)

Example:
Sentence Number: 198 1 -1 LEXIS 
 But the real future of biotechnology lies in addressing the special
 problems faced by farmers in less developed nations.

This is the 198th sentence.  It was classified as 'certain' by the
first annotator, 'uncertain' by the second annotator, and is extracted
from lexis dataset.

The details of the annotation policy are described in the
section 4 ("Hedging to distinguish scientific text: Initial
annotation") of the accompanying paper.
