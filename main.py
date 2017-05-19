import gensim
from nltk.tokenize import word_tokenize

raw_documents = ["Brennan, Instructions and QC Checlist in the folder - M:\DishNetwork\SlingTV\International\n\051617\May 2017Sling International DBS Former Data InstructionsProject Notes:• All suppressions should be done at the address level.• All records should be fully deliverable, total us, both dwelling types.• File layout should include standard fields such as name and address• Each file should be posted with its corresponding NCOA/CASS documentation.• When files are complete, they can be posted to the Sling SFTP and Japs-Olson (Amy, listed as Dish-Japs in the SFTP docs) Please provide the postal paperwork as well.• All files should have the following suppressions:o Dish Current Customerso Dish Do NOT Contactso Sling Current Customers Where No Shipping or Billing Address exists, please suppress at the last name and zip level.o Sling Do Not Contactso Sling Executiveso “Bi-Cyclers” aka people who have signed up for multiple free trials but never converted. File is called: SuppressionFile_BadHouseholds_110215 and is located here: M:\DishNetwork\SlingTV\103015o DECEASED (at the individual level)o Suppress Almquist on Fname and Lnameo Vulgar (On Name)• Please provide a waterfall of suppressions for each file• There should be no duplicates across files. In files of the same language groups, Sling Formers takes priority, DBS Formers is second and prospects are last priority for deduping.DBS Hindi Formers: File Name – 06-SA-01Expected Qty: 17,000Instructions:1) Please combine the following files M:\DishNetwork\SlingTV\International\051617\ Hindi DBS Formers Sling 05_16 and Urdu DBS Formers Sling 05_16 and retain a field that indicates the originating file name in a field called KEYCODE12) Please run through CASS and NCOAa. Please update the address where there is an NCOA hitb. Please only use records that are AABB3) Please suppress out all applicable files as listed aboveLayout should be as follows:ACNT_ID (will be blank)FNAMELNAMEADDRESS 1ADDRESS 2CITYSTATEZIP5ZIP4SPEEDEONID (will be blank)KEYCODE1 (file name)KEYCODE2KEYCODE3KEYCODE4May 2017 Sling International South Asian ProspectsUsing the file here - M:\DishNetwork\SlingTV\International\051617\SouthAsianProspects\ Sling_scoring_southasia_Mar2017.txtPlease select South Asian Records that have 10 mbs + or more download speed based on the broadband chart that Cody has access to, and also fall into the attached SELECT Zips and not in the attached OMITs group• These records should be SFDU and MFDU, have all standard sling suppressions. All suppressions should be done at the address level.• All records should be fully deliverable, total us, both dwelling types.• File layout should include standard fields such as name and address• Each file should be posted with its corresponding NCOA/CASS documentation.• When files are complete, they can be posted to the Sling SFTP and Japs-Olson (Amy, listed as Dish-Japs in the SFTP docs) Please provide the postal paperwork as well.• All files should have the following suppressions:o Dish Current Customerso Dish Do NOT Contactso Sling Current Customers Where No Shipping or Billing Address exists, please suppress at the last name and zip level.o Sling Do Not Contactso Sling Executiveso “Bi-Cyclers” aka people who have signed up for multiple free trials but never converted. File is called: SuppressionFile_BadHouseholds_110215 and is located here: M:\DishNetwork\SlingTV\103015o DECEASED (at the individual level)o Suppress Almquist on Fname and Lnameo Vulgar (on name)o Suppress above listed formers.• Please provide a waterfall of suppressions for each file• There should be no duplicates across files.Please pull top available 300,000 prospects after all of the above is run and output into one file called 06-SA-02. Please make sure to retain decile in KEYCODE1File layout should be as such:ACNT_ID (will be blank)FNAMELNAMEADDRESS 1ADDRESS 2CITYSTATEZIP5ZIP4SPEEDEONIDKEYCODE1 (Decile)KEYCODE2KEYCODE3KEYCODE4All files should be pipe delimited text and posted to japs olson zipped up with their appropriate CASS/NCOA paperwork"]


def get_tokens(text):
    tokens = word_tokenize(text)
    return tokens

gen_docs = [get_tokens(text) for text in raw_documents]

print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
num_words = len(dictionary)
print('Num words in dict: {}'.format(num_words))
for idx, word in dictionary.items():
    print(idx, word)
#
# print(dictionary[7])
# print(dictionary.id2token[7])
#
# bow_doc = dictionary.doc2bow(["Brennan"])
#
# print(bow_doc)

corpus = [dictionary.doc2bow(gen_docs) for gen_docs in gen_docs]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)

print(gen_docs[0])
print(corpus[0])
print(tf_idf[corpus][0])

bow = dictionary.doc2bow(['Brennan'])

