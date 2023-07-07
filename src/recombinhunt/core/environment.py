import pandas as pd
from logging import warning

class Environment:

    def __init__(self, data_dir_path, ignore_lineage=tuple()):
        self.data_dir_path = data_dir_path
        self.lc_df = pd.read_pickle(data_dir_path + "/lc_df.pickle")
        self.c2lp_df = pd.read_pickle(data_dir_path + "/change2lineage_probability.pickle")
        self.cp_df = pd.read_pickle(data_dir_path + "/change_probability.pickle")
        self._c2lp_mutations = set(self.c2lp_df.index)
        self._cp_mutations = set(self.cp_df.index)
        self.x_characterization = None      # is initialized on request
        self.lc_quality_df = pd.read_pickle(data_dir_path + "/lc_quality_df.pickle")

        for l in ignore_lineage:
            try:
                self.lc_df.drop(columns=l.upper(), inplace=True)
                self.c2lp_df.drop(columns=l.upper(), inplace=True)
            except KeyError:
                pass

    def included_lineages(self):
        return self.lc_df.columns[:-1].tolist()

    @staticmethod
    def sequence_nuc_mutations2df(sequence_changes):
        temp_df = pd.DataFrame(index=sequence_changes)
        temp_df['seq_pos'] = temp_df.index.str.split('_').str[0]
        temp_df = temp_df.astype({'seq_pos': int})
        return temp_df

    def make_merged_df(self, seq_df):
        """
        Merge lineage characterization DF and sequence characterization DF in a new DF containing changes from both
        as index and sorted by genomic coordinates. Includes one boolean column for every lineage + one boolean column
        for the sequence (seq_change) + a column of genomic coordinates (merg_pos).
        :param seq_df: sequence DF as returned from Data.sequence_nuc_mutations2df
        :return: a DF
        """
        merged_df = pd.merge(self.lc_df, seq_df, how='outer', left_index=True, right_index=True)  # merge on change name
        merged_df['seq_change'] = merged_df.seq_pos.notnull()  # flag for changes present in sequence
        merged_df['merg_pos'] = merged_df.lc_pos.fillna(
            merged_df.seq_pos)  # new column merg_pos from lc_pos + seq_pos
        merged_df = merged_df.astype({'merg_pos': int})
        merged_df = merged_df.drop(['lc_pos', 'seq_pos'], axis=1).fillna(False)
        merged_df = merged_df.sort_values(by='merg_pos')
        return merged_df

    def probabilities(self, seq_df):
        changes = set(self.lc_df.index.to_list() + seq_df.index.to_list())

        p_merged_df = pd.merge(
            self.c2lp_df.loc[list(changes & self._c2lp_mutations)],  # the loc is to reduce the size of the joined tables,
            seq_df,
            how='outer', left_index=True, right_index=True)  # merge on change name
        p_merged_df['seq_change'] = p_merged_df.seq_pos.notnull()  # flag for changes present in sequence
        p_merged_df['merg_pos'] = p_merged_df.pos.fillna(p_merged_df.seq_pos)  # new column merg_pos from lc_pos + seq_pos
        p_merged_df = p_merged_df.astype({'merg_pos': int})
        p_merged_df = p_merged_df.drop(['pos', 'seq_pos'], axis=1).fillna(0.0)
        p_merged_df = p_merged_df.sort_values(by='merg_pos')

        cp_df = pd.merge(
            self.cp_df.loc[list(changes & self._cp_mutations)],     # loc to reduce size of joined tables
            seq_df,
            how='outer', left_index=True, right_index=True)\
        .fillna(0.0)
        cp_df.pos = cp_df.pos.fillna(cp_df.seq_pos)
        cp_df = cp_df.drop(columns=['seq_pos']).astype({'pos': int}).sort_values(by='pos').drop(columns=['pos'])
        change_probabilities = cp_df.values.reshape(cp_df.shape[0])

        return p_merged_df, change_probabilities

    def change_probability(self, change):
        try:
            return self.cp_df.at[change, 'probability']
        except KeyError:
            return 0.0

    def __getattribute__(self, item):
        if item == "x_characterization":
            self.__setattr__("x_characterization", self.__get_x_characterization())
        return super(Environment, self).__getattribute__(item)

    def __get_x_characterization(self):
        try:
            xlc_df = pd.read_pickle(self.data_dir_path + "/lc_df_with_X.pickle")
        except FileNotFoundError:
            raise FileNotFoundError(f"Characterization for recombinant lineages not available in the current "
                                    f"environment {self.data_dir_path}")
        xlc_df = xlc_df[[l for l in xlc_df.columns if l.startswith('X')]]
        return xlc_df

    def x_characterizing_nuc_mutations(self, x_lineage_name):
        if x_lineage_name.upper() not in self.x_characterization.columns:
            raise KeyError(f"Lineage characterization for {x_lineage_name} is not available in the environment "
                           f"{self.data_dir_path}")
        return self.x_characterization.index[self.x_characterization[x_lineage_name.upper()]].values.tolist()

    def number_of_sequences_of_lineage(self, lineage_name):
        try:
            return self.lc_quality_df.at['num', lineage_name.upper()]
        except KeyError:
            return 0

#
# class Environment:
#
#     def __init__(self, data_dir_path, ignore_lineage=tuple()):
#         self.data_dir_path = data_dir_path
#         self.lc_df = pd.read_pickle(data_dir_path + "/lc_df.pickle")
#         self.c2lp_df = pd.read_pickle(data_dir_path + "/change2lineage_probability.pickle")
#         self.cp_df = pd.read_pickle(data_dir_path + "/change_probability.pickle")
#         # self._c2lp_mutations = set(self.c2lp_df.index)
#         self.x_characterization = None      # is initialized on request
#         self.lc_quality_df = pd.read_pickle(data_dir_path + "/lc_quality_df.pickle")
#
#         for l in ignore_lineage:
#             try:
#                 self.lc_df.drop(columns=l.upper(), inplace=True)
#                 self.c2lp_df.drop(columns=l.upper(), inplace=True)
#             except KeyError:
#                 pass
#
#     def included_lineages(self):
#         return self.lc_df.columns[:-1].tolist()
#
#     @staticmethod
#     def sequence_nuc_mutations2df(sequence_changes):
#         temp_df = pd.DataFrame(index=sequence_changes)
#         temp_df['seq_pos'] = temp_df.index.str.split('_').str[0]
#         temp_df = temp_df.astype({'seq_pos': int})
#         return temp_df
#
#     # def sequence_nuc_mutations2df(self, sequence_changes):
#     #     temp_df = pd.DataFrame(index=list(set(sequence_changes) & self._c2lp_mutations))
#     #     temp_df['seq_pos'] = temp_df.index.str.split('_').str[0]
#     #     temp_df = temp_df.astype({'seq_pos': int})
#     #     return temp_df
#
#
#     def make_merged_df(self, seq_df):
#         """
#         Merge lineage characterization DF and sequence characterization DF in a new DF containing changes from both
#         as index and sorted by genomic coordinates. Includes one boolean column for every lineage + one boolean column
#         for the sequence (seq_change) + a column of genomic coordinates (merg_pos).
#         :param seq_df: sequence DF as returned from Data.sequence_nuc_mutations2df
#         :return: a DF
#         """
#         merged_df = pd.merge(self.lc_df, seq_df, how='outer', left_index=True, right_index=True)  # merge on change name
#         merged_df['seq_change'] = merged_df.seq_pos.notnull()  # flag for changes present in sequence
#         merged_df['merg_pos'] = merged_df.lc_pos.fillna(
#             merged_df.seq_pos)  # new column merg_pos from lc_pos + seq_pos
#         merged_df = merged_df.astype({'merg_pos': int})
#         merged_df = merged_df.drop(['lc_pos', 'seq_pos'], axis=1).fillna(False)
#         merged_df = merged_df.sort_values(by='merg_pos')
#         return merged_df
#
#     def probabilities(self, seq_df):
#         included_changes = list(set(self.lc_df.index.to_list() + seq_df.index.to_list()))
#         p_merged_df = pd.merge(
#             self.c2lp_df.loc[included_changes],  # the loc is to reduce the size of the joined tables
#             seq_df,
#             how='outer', left_index=True, right_index=True)  # merge on change name
#         p_merged_df['seq_change'] = p_merged_df.seq_pos.notnull()  # flag for changes present in sequence
#         p_merged_df['merg_pos'] = p_merged_df.pos.fillna(p_merged_df.seq_pos)  # new column merg_pos from lc_pos + seq_pos
#         p_merged_df = p_merged_df.astype({'merg_pos': int})
#         p_merged_df = p_merged_df.drop(['pos', 'seq_pos'], axis=1)
#         p_merged_df = p_merged_df.sort_values(by='merg_pos')
#
#         change_probabilities = (self.cp_df.loc[p_merged_df.index]
#                 .drop(columns=['pos'])
#                 .values.reshape(p_merged_df.shape[0]))
#
#         return p_merged_df, change_probabilities
#
#     # def probabilities(self, seq_df):
#     #     included_known_changes = list(set(self.lc_df.index.to_list() + seq_df.index.to_list()) & self._c2lp_mutations)
#     #
#     #     p_merged_df = pd.merge(
#     #         self.c2lp_df.loc[included_known_changes],  # the loc is to reduce the size of the joined tables,
#     #         seq_df,
#     #         how='outer', left_index=True, right_index=True)  # merge on change name
#     #     p_merged_df['seq_change'] = p_merged_df.seq_pos.notnull()  # flag for changes present in sequence
#     #     p_merged_df['merg_pos'] = p_merged_df.pos.fillna(p_merged_df.seq_pos)  # new column merg_pos from lc_pos + seq_pos
#     #     p_merged_df = p_merged_df.astype({'merg_pos': int})
#     #     p_merged_df = p_merged_df.drop(['pos', 'seq_pos'], axis=1).fillna(0.0)
#     #     p_merged_df = p_merged_df.sort_values(by='merg_pos')
#     #
#     #     change_probabilities = (
#     #         pd.merge(
#     #             self.cp_df.loc[included_known_changes],     # loc to reduce size of joined tables
#     #             seq_df,
#     #             how='outer', left_index=True, right_index=True)
#     #         .drop(columns=['pos', 'seq_pos']).fillna(0.0)
#     #         .values.reshape(p_merged_df.shape[0]))
#     #
#     #     return p_merged_df, change_probabilities
#
#     def __getattribute__(self, item):
#         if item == "x_characterization":
#             self.__setattr__("x_characterization", self.__get_x_characterization())
#         return super(Environment, self).__getattribute__(item)
#
#     def __get_x_characterization(self):
#         try:
#             xlc_df = pd.read_pickle(self.data_dir_path + "/lc_df_with_X.pickle")
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Characterization for recombinant lineages not available in the current "
#                                     f"environment {self.data_dir_path}")
#         xlc_df = xlc_df[[l for l in xlc_df.columns if l.startswith('X')]]
#         return xlc_df
#
#     def x_characterizing_nuc_mutations(self, x_lineage_name):
#         if x_lineage_name.upper() not in self.x_characterization.columns:
#             raise KeyError(f"Lineage characterization for {x_lineage_name} is not available in the environment "
#                            f"{self.data_dir_path}")
#         return self.x_characterization.index[self.x_characterization[x_lineage_name.upper()]].values.tolist()
#
#     def number_of_sequences_of_lineage(self, lineage_name):
#         try:
#             return self.lc_quality_df.at['num', lineage_name.upper()]
#         except KeyError:
#             return 0


# class CharacterizationRecombinant:
#     def __init__(self, data_dir_path):
#         self.data_dir_path = data_dir_path
#
#         xlc_df = pd.read_pickle(self.data_dir_path + "/lc_df_with_X.pickle")
#         xlc_df = xlc_df[[l for l in xlc_df.columns if l.startswith('X')]]
#         self.x_characterization = xlc_df
#
#     def x_characterizing_nuc_mutations(self, x_lineage_name):
#         if x_lineage_name.upper() not in self.x_characterization.columns:
#             raise KeyError(f"Lineage characterization for {x_lineage_name} is not available in the environment "
#                            f"{self.data_dir_path}")
#         return self.x_characterization.index[self.x_characterization[x_lineage_name.upper()]].values.tolist()
