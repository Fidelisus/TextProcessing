import pandas as pd
import numpy as np
import scipy.stats as stats
from operator import itemgetter

class JudgementAggregation:
    def __init__(self, main_folder):
        # count of the times that user agreed on the score with someone else
        self.users_trust = {}
        # count of the times that user annotated, given there was also someone else annotating
        # it will be used to normalize users_trust
        self.users_annotations_count = {}
        self.main_folder = main_folder
        
    def load_data(self):
        # paths to relevant datasets
        config_p1 = {
            "docs": self.main_folder + "fira-22.documents.tsv",
            "judgements": self.main_folder + "fira-22.judgements-anonymized.tsv",
            "queries": self.main_folder + "fira-22.queries.tsv",
        }

        # Document text with id
        documents = pd.read_csv(config_p1["docs"],sep='\t', header= 0, names= ["documentId", "documentText"], index_col="documentId")
        # relevance score given to query>document pair by a particular user and some additional info
        judgements_p1 = pd.read_csv(config_p1["judgements"], sep='\t')
        # Query text with id
        queries = pd.read_csv(config_p1["queries"],sep='\t', header= 0, names= ["queryId", "queryText"], index_col="queryId")

        self.judgements_p1 = judgements_p1.join(documents, how="inner", on="documentId").join(queries, how="inner", on="queryId")
        self.judgements_p1["queryIteration"] = "Q0"

        # transform relevanceLevels to int
        self.judgements_p1["relevanceLevel"] = self.judgements_p1["relevanceLevel"].replace(
            ["0_NOT_RELEVANT", "1_TOPIC_RELEVANT_DOES_NOT_ANSWER", "2_GOOD_ANSWER", "3_PERFECT_ANSWER"], [0, 1, 2, 3])

    def calculate_user_trust(self, levels):
        def increment_or_create_user_count(key, dictionary, value=1):
            if key in dictionary:
                dictionary[key] += value
            else:
                dictionary[key] = value

        if levels.shape[0] == 1:
            return None

        users = levels[["userId", "relevanceLevel"]].set_index('userId').T.to_dict('list')

        for user in users.keys():
            increment_or_create_user_count(user, self.users_annotations_count)

            users_that_agree = 0
            for relevance in users.values():
                # we count number of the times that user agreed on the score with someone else
                # it means that if 3 users agreed on the grade, each of them gets 2 points
                if relevance == users[user]:
                    users_that_agree += 1
            increment_or_create_user_count(user, self.users_trust, users_that_agree-1)

    def unify_relevance_levels(self, df):
        def get_max_distance_between_relevances(relevances):
            sorted_relevances = sorted(relevances)
            return sorted_relevances[-1] - sorted_relevances[0]

        # 0) When only 1 user rates
        if df.shape[0] == 1:
            return df

        user_relevance_dict = df[["userId", "relevanceLevel"]].set_index('userId').T.to_dict('list')

        # get relevances from the dictionary
        relevances = [item for sublist in list(user_relevance_dict.values()) for item in sublist]
        max_distance_between_relevances = get_max_distance_between_relevances(relevances)

        users = list(user_relevance_dict.keys())
        # 1) If there are only two scores and they are close to each other
        if df.shape[0] == 2 and max_distance_between_relevances <= 1.0:
            # a) if users agree
            if user_relevance_dict[users[0]] == user_relevance_dict[users[1]]:
                df["relevanceLevel"] = user_relevance_dict[users[0]][0]
                return df.iloc[[0]]
            # b if users disagree then we trust one with higher users_trust_normalized
            elif self.users_trust_normalized[users[0]] >= self.users_trust_normalized[users[1]]:
                df["relevanceLevel"] = user_relevance_dict[users[0]][0]
                return df.iloc[[0]]
            else:
                df["relevanceLevel"] = user_relevance_dict[users[1]][0]
                return df.iloc[[0]]
        # 2) If there are only two relevances and they are far apart we don't trust it
        elif df.shape[0] == 2 and max_distance_between_relevances > 1.0:
            return df[:0]
        # 3) If there are 3 scores
        elif df.shape[0] == 3:
            # a) If everyone agrees then it is easy
            if len(set(relevances)) == 1:
                df["relevanceLevel"] = user_relevance_dict[users[0]][0]
                return df.iloc[[0]]
            # b) If the max distance between scores is ==3 then we don't trust it
            elif max_distance_between_relevances == 3:
                return df[:0]
            # c) if the max distance between scores is <=2 then we calculate weighted mean
            elif max_distance_between_relevances <= 2:
                weighted_sum = 0.0
                denominator = 0.0
                for user in users:
                    weighted_sum += user_relevance_dict[user][0]*self.users_trust_normalized[user]
                    denominator += self.users_trust_normalized[user]
                if denominator != 0.0:
                    df["relevanceLevel"] = round(weighted_sum/denominator)
                    return df.iloc[[0]]
            
    def aggregate(self, file_path=None):
        self.load_data()
        
        # Filter judgements 
        self.judgements_p1 = self.judgements_p1[self.judgements_p1["durationUsedToJudgeMs"] >= 2000]
        
        self.judgements_p1.groupby(["queryId", "documentId"]).apply(self.calculate_user_trust)

        # normalize user_trust by number of annotations of each user
        # the normalized value can be bigger than 1, as we count each time a user agrees with other
        # so i.e. if 3 users agreed on the grade, each of them gets 2 points
        std = np.std(list(self.users_trust.values()))
        self.users_trust_normalized = {k: v / self.users_annotations_count[k] for k, v in self.users_trust.items()}
        
        judgements_p1_transformed = self.judgements_p1.groupby(["queryId", "documentId"]).apply(
            self.unify_relevance_levels).drop(["id", "queryId", "documentId"], axis = 1
        ).reset_index().drop(["level_2"], axis = 1)
        
        if file_path:
            judgements_p1_transformed.to_csv(file_path, index=False, header=False, sep='\t',
                                 columns=["queryId", "queryIteration", "documentId", "relevanceLevel",
                                "queryText" ,"documentText"])
        
        return judgements_p1_transformed
