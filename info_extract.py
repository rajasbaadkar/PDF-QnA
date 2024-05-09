import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  

test_transcript = "Good morning! This is Emily from Dream Homes. How may I assist you today? Hi Emily, I'm James. I'm looking for a new place, preferably a 3-bedroom house or a spacious 2-bedroom apartment. Great choice, James! Are you particular about the location? Yes, I'm aiming for something near downtown. Maybe around the Riverfront area? Absolutely, Riverfront properties are in high demand. We have a few options available. How about your budget? I'm flexible but looking at something between $500,000 to $700,000. That's reasonable for the area. We have a stunning 3-bedroom condo with a panoramic river view for $650,000. Sounds interesting! Is it a new building? Yes, it’s relatively new, just three years old. The amenities are impressive too – gym, pool, and a private park. Wonderful! How about the nearby schools? There's a reputable school just a few blocks away, perfect for families. That's crucial. I'd love to schedule a visit soon. Of course, James! I'll email you the details along with available viewing times. Thanks! Looking forward to it. My pleasure. Have a great day, James!"
# transcript = "Hello. Hello. Yes. Hi sir, this is Rohit from L&T Realty. We have just received your enquiry on one of our... Yeah, yeah, yeah. Yeah sir. So, if you could just help me with your basic requirements sir, what we are looking for? See, I am looking for either a large 2BHK or a 3BHK. Okay, great sir. Is there any considerable timeline we are, you know, prior to it? I mean, are you under construction or are you ready to move in? No, ready to move in. Okay. In ready to move in, sir, we have some limited inventories in Emerald Isle, which is, you know, with the carpet areas of 640 to 808. So, for now, sir, we will have around 700 plus inventories available, sir. Okay. So, to be precise, you will be having an option of 650 and 734 and 640 and 808, sir. 808 is 2BHK. Yes sir, yes sir. All these are your Rera carpets, sir. So, pricing wise, if I talk, we are within the range of, you know, starting from 2.30 to all the way to 2.85 onwards, depending on, you know, how the carpet areas we are preferring. No, like for example... For example, if I ask you for the 808. Yes, sir. Then how would it, how much would it cost? Yes, sir. For 808, it will be starting from 2.85. So, then again, just to make a clearer statement, sir, your 2.85 includes your agreement value plus your stamp duty registration charges, your advanced maintenance and your car park also. So, the only thing would be additional and subjective to a particular unit is your floor rise, sir. Right, sir. Right, sir. And also, I would like to mention that there are, you know, different types of payment structures more, you know, keen towards flexibility. So, we have a subvention offer of 2080 where you make a partial payment of 20% at the time of booking and you have to pay 80% of your post possession. Apart from that... I'm talking about the ready to move in. Right, sir. Right, sir. So, if you are looking for an option of 2023 or something, then you can try for that, sir. Just, just giving an understanding of what are the, you know, the availabilities and the options, like, if you prefer anything other than ready to move in, so, sir. Right, sir. But it went a little higher. Pardon me, sir. Could you repeat yourself? I'm saying it went a little higher. What I heard is not just much price, 2.85. Sir, if you look at the project, then we have a land parcel of 19 acres. So, in that also, we are not making it very much crowded, sir. We are keeping the elevation only till 18 storey. All our towers are going to be of 18 storey. And if you compare open spaces, then you will also get to see 9 to 10 acres of open land which is dedicated for amenities and open spaces. So, we are not actually, you know, crowding the place. That is no problem. See, I am already staying in Barol, Lhasaan Toys, I stay in Lhasaan Toys. You are coming from Mumbai or where? So, it, pardon me, sir. I don't understand. Where are you from? I am talking from Powai, sir. You are from Powai, right? I am actually calling from, right, sir. So, this true caller is coming from Bangalore. That, sir, is for the dialer, actually. So, this true caller is yours. Yes, sir. So, these numbers are the numbers purchased by our company. Okay, okay. See, I stay in Lhasaan Toys, sir. You know Lhasaan Toys, sir? Right, sir. I stay there. Yes, yes, sir. Which is in Andheri, I believe, right, sir? The one that comes towards Andheri, Barol. Yes, Lhasaan Toys, sir. Right, sir. It is also a good, very good society. It is a 14-acre society. Right, sir. All your work is done here. Right, sir. Right, sir. So, the facilities and all, greenery, well, whatever is there. Leave it, sir. I understand, sir. I have a free car park here. Right, sir. I also have a free car park here. That is no problem. Leave that. Right, sir. Now tell me, I need a 2BHK for self-use, not an investment. I understand, sir. I would like to shift from here. So, Rohit, I need an east-facing flat on the highest floor. Okay. How many carpets do you need? Check it and tell me. Right, sir. So, how is it, sir? To give you an understanding, if you are talking about the option of RTMIs, then we have currently at the podium level, that is, your podium level is around the height of the sixth or seventh floor. So, we have these 2BHKs available at this elevation only, sir. Apart from that, all the units have been sold out and… You won't even see the pavali lake in that elevation. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. No, sir. What is this pavali lake, Sir? Is there any other option available? No, sir. No, sir. No, sir. No, sir. No, sir. No, no, sir. No, sir. No, sir. Yes, sir. No, sir. No, sir. No, sir. and 22nd floor in the Atheist and Iora also so there you have the option of partial lake view but then again the possession timeline that is quite different from this so that is uh in the possession timeline of june 25th december sorry june 2025 so ready position east facing flag data so uh what i can do is sir you know open view means internal view now open view so uh just for an understanding sir uh it would be better you know if you can come down at the site and have a discussion with them because real-time availability shifts so so what we have here is the podium level options for 2bh case of after 650 734 640 in fact 808 so if in case there are any you know cancellations so we can uh you know any any time give you a call but for that you will have to be one of our visited customers so we actually follow a protocol where we give the first preference to visited customers so you will have an opportunity sir if at all you know if the case is not there yet i will visit you in a couple of days time for five days not an issue not an issue in the meantime what i do is i share some details with you just so just so you know you can go through a floor plan through go go through so once you come to the sales gallery sir you as soon as your requirement remains according to your best suitability you can check the availability of inventories so that won't be a problem yes sir i'm just sharing the details with you okay anything else i can help you with sir no thank you all right thank you so much have a wonderful day"
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=350,
                        chunk_overlap=100,
                        length_function=len
                )
        
chunks = text_splitter.split_text(test_transcript)
vectorstore = FAISS.from_texts(chunks,embedding = embeddings)

question = "What is the budget of the client? Give answer in key:value pair."

docs = vectorstore.similarity_search(query = question, k=3)     #k is context window
chat_model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

chain = load_qa_chain(llm=chat_model,chain_type="stuff")
with get_openai_callback() as cb:
    response = chain.run(input_documents=docs,question=question)
    # print(cb)
print(response)