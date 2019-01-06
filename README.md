# sc2-machine-learning
A bot using TensorFlow and CUDA to utilize machine learning in playing StarCraft 2


Written for use in Python 3.4 and upwards, probably doesn't work for earlier versions. 

# TensorFlow
This bot uses TensorFlow to utilize the machine learning capabilities it offers, to make the bot more human-like instead of being a pure state machine. The model it is using isn't the most optimized one, but it works fine for using it to play against the standard StarCraft 2 bots.

# PySC2
This bot uses the framework PySC2 (https://github.com/deepmind/pysc2) for easier use of game resources.

# Use
Open either the BlankBot.py or BlankBotV2.py depending on which is the version that is located in the repository, and change the "SC2PATH"-variable that is at the quite top of the script to wherever you're StarCraft 2 is installed. You can download the headless version for free at https://github.com/Blizzard/s2client-proto. Remember to also download some map packs to use with it!

When you've decided on a map you want to make the bot play, change the function input variable at the bottom of the script in the "run_game" function to whatever the maps name is. Currently it is set to "CatalystLE", which I think is a pretty good map for the bot, it seems to win approximately 60% of the time against medium AI in that one.
