from TwitterSearch import *
import re
from collections import Counter
import os

try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
#    tso.set_keywords(['Guttenberg', 'Doktorarbeit']) # let's define all words we would like to have a look for
    tso.set_keywords(['low trust', 'GMO'])
#    tso.set_language('de') # we want to see German tweets only
    tso.set_language('en')
#    tso.set_include_entities(False) # and don't give us all those entity information
    tso.set_include_entities(True)
    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
        consumer_key = 'SQs0N4i8MLiZZf5j4KdjLkFNt',
        consumer_secret = 'oNMf3qExMdxR9BASmGmEVQuPfA4K1j1RAhaT3CkB1Vfx3bwGi6',
        access_token = '3286257269-2GxMXtboCXCUYBufiuY75FXqEfP44XmGU2IpjV4',
        access_token_secret = 'zSjx5KhP7HOqEq77xAc2JUDwldDzLBcMDD93fcpHnA41I'
     )
    
     # this is where the fun actually starts :)
    for tweet in ts.search_tweets_iterable(tso):     
          
          output = open('DataSet_con_12.txt','a',encoding='utf-8')        
         # Pull out various data from the tweets
         # output.write('%s,%s,%s\n' % (tweet['user']['screen_name'],tweet['user']['name'], tweet['text'],))
          tweetRemove = tweet['text']
          tweetRemove = ' '.join(re.sub("(RT)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetRemove).split())
         # check file is empty or not, if empty then insert first line
          if os.stat('DataSet_con_12.txt').st_size == 0:
                  old_text = tweetRemove
                  output.write('%s\n' % (tweetRemove,))
                  print('saved_text==null')
         #let current tweets compare to previous line, if the same then move on, if not save into file
          else:
                  if Counter(tweetRemove) != Counter(old_text):
                      old_text = tweetRemove
                      output.write('%s\n' % (tweetRemove,))
                  
          output.close()       
#         print( '@%s name is %s tweeted: %s' % ( tweet['user']['screen_name'],tweet['user']['name'], tweet['text'].encode('utf8','ignore'), ) )

except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)
