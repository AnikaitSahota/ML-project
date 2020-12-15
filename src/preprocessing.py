import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.decomposition import PCA , TruncatedSVD



class preprocessor() :
	"""class to read the dataset and clean it for for furthur processsing
	"""
	def __init__(self , DATASET_PATH = 'data/dataset_') :
		"""constructor for the preprocessord class. It specifies the path to dataset

		Parameters
		----------
		DATASET_PATH : str, optional
			path to the dataset, by default 'data/dataset_' for all splitted files
		"""
		# If you want to use the single file i.e.DATASET_PATH = 'data/training.1600000.processed.noemoticon.csv'
		# then self.df = self.read_dataset(DATASET_PATH, join_flag = False)
		self.df = self.read_dataset(DATASET_PATH)			# setter for the dataset

	def read_dataset(self , DATASET_PATH , join_flag = True , ext = '.csv') :
		"""function to extract the dataset (dataframe) from the specified file path

		Parameters
		----------
		DATASET_PATH : string
			path to dataset
		join_flag : boolean
			flag to indicate whether to extract data from splited files or a single file
		ext : string
			extenstion of splitted files (only to be used when join_flag = True), by default '.csv.'

		Returns
		-------
		pandas dataframe
			data extracted from file at specified path
		"""
		columns = [ 'target' , 'ID' , 'date' , 'flag' , 'user' , 'text' ]			# naming the columns
		if(join_flag) :							# joining dataset
			frames = []							# dataset was divided by 'split -C 20971520 -d training.1600000.processed.noemoticon.csv --additional-suffix=.csv dataset_' where 20971520 B = 20 MB
			for i in range(12) :
				frames.append(pd.read_csv(DATASET_PATH + str(i).zfill(2) + ext , encoding = "ISO-8859-1" , header=None , names=columns)) 
			df = pd.concat(frames , ignore_index=True)
		else :
			df = pd.read_csv(DATASET_PATH , encoding = "ISO-8859-1" , header= None , names = columns)

		df.loc[df['target'] == 4, 'target'] = 1                 # changing the target value from 4 to 1
		return df												# returning dataset

	def __get_day(self,x) :
		"""function to tell the day (int) of tweat based upon date-time

		Parameters
		----------
		x : string-like
			Date-time of tweat

		Returns
		-------
		int
			number associated with the day i.e., (0: Monday, 1: Tuesday, 2: Wednessday, 3: Thursday, 4: Friday, 5: Saturday, 6: Sunday, -1: None)
		"""
		days = ['Mon' , 'Tue' , 'Wed' , 'Thu' , 'Fri' , 'Sat' , 'Sun']
		d = x[:3]														# sliciing the day string from date-time string
		if(d in days) :
			return days.index(d)
		return -1														# -1 if day is not known

	def get_user_resolved(self) :
		"""function to make usefull feature out of user feature (which contain usernames)
		We have appliad one-hot-encoding to extract the uniqueness and repeatition of a user.
		Since the there were too many unique users (i.e. high dimensional data), hence dimension reduction is done

		Returns
		-------
		numpy 2D array 
			Array contains resolved features of user column
		"""
		user_ndarr = self.df['user'].to_numpy().reshape(-1,1)
		encoder = OneHotEncoder()
		encoder = encoder.fit(user_ndarr)
		hot_coded = encoder.transform(user_ndarr)						# hot_coded is scipy.sparse.csr.csr_matrix, which is a memory efficent way of storing 1-hot-coded matrix

		tsvd = TruncatedSVD(n_components= 50)
		return tsvd.fit(hot_coded).transform(hot_coded)
		# TODO : what to choose TVSD and PCA --> also add this docstring 'by using PCA'
		# dim_red = PCA(n_components=2)
		# dim_red.fit(hot_coded)
		# return dim_red.transform(hot_coded)

	def remove_pattern(self , text , pattern):
		"""fucntion to clean the tweats for furtur processing.
		Here we are removing the specified pattern from text

		Parameters
		----------
		text : string
			the text of tweat
		pattern : string
			the pattern to be removed

		Returns
		-------
		string
			cleaned tweat
		"""
		r = re.findall(pattern,text)			# finds the pattern i.e @user and puts it in a list for further task

		for i in r:
			text = re.sub(i,"",text)
		
		return text

	def preprocess(self) :
		"""function to preprocess the dataframe and return dependent (X) and independent (y)

		Returns
		-------
		X : numpy 2D array
			it is the array of features
		y : numpy 1D array
			it is the array of labels
		"""

		day = self.df.date.apply(lambda x : self.__get_day(x))
		self.df['date'] = pd.to_datetime(self.df['date'])
		date = self.df.date.apply(lambda x : x.day)
		month = self.df.date.apply(lambda x : x.month)
		year = self.df.date.apply(lambda x : x.year)
		time_in_minutes = self.df.date.apply(lambda x : x.minute + x.hour * 60)

		usr = self.get_user_resolved()

		self.df['Tidy_Tweets'] = np.vectorize(self.remove_pattern)(self.df['text'], "@[\w]*")
		self.df['Tidy_Tweets'] = self.df['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
		self.df['Tidy_Tweets'] = self.df['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
		# Bag of Words
		bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
		# bag-of-words feature matrix
		bow = bow_vectorizer.fit_transform(self.df['Tidy_Tweets'])
		# df_bow = pd.DataFrame(bow.todense())
		# train_bow = bow
		tsvd = TruncatedSVD(n_components=200)
		tweets_resolved = tsvd.fit_transform(bow)
		usr = np.append(usr , tweets_resolved , axis=1)

		X = pd.concat ([ day , date , month , year , time_in_minutes ] , axis = 1).to_numpy()
		X = np.append(X,usr , axis=1)
		# X = 1
		y = self.df['target'].to_numpy()

		return X , y