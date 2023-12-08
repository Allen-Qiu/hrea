# using llama to build a multiclass classifier on Yelp dataset

import torch
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import accelerate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import re 

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
print("Using device", device)

review_file = 'yelp-reviews.json'
fin         = open(review_file)
lines       = fin.readlines()
review_list = list()
rating_list = list()

for line in lines:
    dic = json.loads(line)
    review_list.append(dic['text'].lower())
    rating_list.append(dic['stars'])
fin.close()

checkpoint = '/root/autodl-tmp/llama2-7b'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device)

fout =open('result.txt','tw')

k = 0
for review, rating in zip(review_list, rating_list):
    # if index%100==0: print(index)
    # print(index)
    if not isinstance(review, str):
        review=3
    review = review.replace('\n', ' ')
    # tid   = row['tid']
    # if tid >21184:break
    # print(f"{tid:} {tweet}")
    prompts = f"""###
A review: My girlfriend and I stayed here for 3 nights and loved it. The location of this hotel and very decent price makes this an amazing deal. When you walk out the front door Scott Monument and Princes street are right in front of you, Edinburgh Castle and the Royal Mile is a 2 minute walk via a close right around the corner, and there are so many hidden gems nearby including Calton Hill and the newly opened Arches that made this location incredible.\n\nThe hotel itself was also very nice with a reasonably priced bar, very considerate staff, and small but comfortable rooms with excellent bathrooms and showers. Only two minor complaints are no telephones in room for room service (not a huge deal for us) and no AC in the room, but they have huge windows which can be fully opened. The staff were incredible though, letting us borrow umbrellas for the rain, giving us maps and directions, and also when we had lost our only UK adapter for charging our phones gave us a very fancy one for free.\n\nI would highly recommend this hotel to friends, and when I return to Edinburgh (which I most definitely will) I will be staying here without any hesitation.
Action: Please choose a star rating from the set [1, 2, 3, 4, 5] for the review.
rating: 5
###
A review: If you need an inexpensive place to stay for a night or two then you may consider this place but for a longer stay I'd recommend somewhere with better amenities. \n\nPros:\nGreat location- you're right by the train station, central location to get to old town and new town, and right by sight seeing his tours. Food, bars, and shopping all within walking distance. Location, location, location.\nVery clean and very good maid service\n\nCons:\nTiny rooms \nUncomfortable bed \nAbsolutely no amenities \nNo phone in room \nNo wardrobe \n\nWas given a lot of attitude about me and my husband sharing a room which was quite strange and we were charged 15 pounds more for double occupancy not sure why that matters I felt like it was a money grab. It was just handled in a kind of odd manner to me... \n\nIf you book this hotel all you get is a bed, desk, and a bathroom. It isn't awful but know what you're getting into.
Action: Please choose a star rating from the set [1, 2, 3, 4, 5] for the review.
rating: 3
###
A review: Location is everything and this hotel has it! The reception is inviting and open 24 hours. They are very helpful and have a lot of patience answering all my questions about where to go etc. there is also a lounge open 24 hours with snack-type food. Breakfast is continental-style so if you want heartier fare look elsewhere though you don't have to go far. The bus and train stations are right across the street so it's easy access to the airport or anywhere else you may want to go. Turn uphill to old town or cross the bridge to new town. The room with a view i got was spacious and comfortable though it's a bit of a maze to find it-just follow the signs. The windows are double paned so the room is quiet plus i was on the 5th floor which helps. It's a bit pricey but still one of the best values i found!
Action: Please choose a star rating from the set [1, 2, 3, 4, 5] for the review.
rating: 4
###
A review: This place is horrible, we were so excited to try it since I got a gift card for my birthday. We went in an ordered are whole meal and they did not except are gift card, because their system was down. Unacceptable, this would have been so helpful if we would have known this prior!!
Action: Please choose a star rating from the set [1, 2, 3, 4, 5] for the review.
rating: 1
###
A review: I had the garlic ginger broccoli chicken and it was not very good. The broccoli was hardly cooked and the sauce was way to sweet. Everything else was great. I will give them a few more tries before I write them off as another crappy Asian restaurant in Surprise.
Action: Please choose a star rating from the set [1, 2, 3, 4, 5] for the review.
rating: 2
###
A tweet: {review}
Action: Please choose a star rating from the set [1, 2, 3, 4, 5] for the review.
Result:"""
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids

    generation_output = model.generate(
        input_ids=input_ids.to(device), max_new_tokens=10
    )
    res = tokenizer.decode(generation_output[0])
    # print(res)
    rlist = res.split("###")
    # record={}
    # record['tid'] = tid
    temp = rlist[-2].strip()
    temp = temp.split("\n")[-1]
    tag = re.split('Result:', temp)
    tag = tag[-1].strip()
    # print(f"{rating} {tag}")
    fout.write(f"{rating} {tag}\n")
    # if k>10: break
    # k += 1

fout.close()