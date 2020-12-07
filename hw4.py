import sys
import pickle as pkl
import array
import os
import timeit
import contextlib
import nltk
import re
import heapq

# try:
#     os.mkdir('output_dir')
# except FileExistsError:
#     pass
# try:
#     os.mkdir('tmp')
# except FileExistsError:
#     pass
#
# sorted(os.listdir('pal-data'))
#
# var = sorted(os.listdir('pal-data/0'))[:10]
#
# print(var)

# with open('pal-data/0/3dradiology.stanford.edu_', 'r') as f:
#     print(f.read())

toy_dir = 'toy-data'


class IdMap:
    """Helper class to store a mapping from strings to ids."""

    def __init__(self):
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Return number of terms stored in the IdMap"""
        return len(self.id_to_str)

    def _get_str(self, i):
        """Returns the string corresponding to a given id (`i`)."""
        # Begin your code
        if i < len(self.id_to_str):
            return self.id_to_str[i]
        else:
            return ''
        # End your code

    def _get_id(self, s):
        """Returns the id corresponding to a string (`s`).
        If `s` is not in the IdMap yet, then assigns a new id and returns the new id.
        """
        # Begin your code
        if s not in self.str_to_id.keys():
            idx = len(self.id_to_str)
            self.id_to_str.append(s)
            self.str_to_id[s] = idx
            return idx
        else:
            return self.str_to_id[s]
        # End your code

    def __getitem__(self, key):
        """If `key` is a integer, use _get_str;
           If `key` is a string, use _get_id;"""
        if type(key) is int:
            return self._get_str(key)
        elif type(key) is str:
            return self._get_id(key)
        else:
            raise TypeError


# # Begin your test code for IdMap
# testIdMap = IdMap()
# assert testIdMap['a'] == 0, "Unable to add a new string to the IdMap"
# assert testIdMap['bcd'] == 1, "Unable to add a new string to the IdMap"
# assert testIdMap['a'] == 0, "Unable to retrieve the id of an existing string"
# assert testIdMap[1] == 'bcd', "Unable to retrive the string corresponding to a                                given id"
# try:
#     testIdMap[2]
# except IndexError as e:
#     assert True, "Doesn't throw an IndexError for out of range numeric ids"
# assert len(testIdMap) == 2
# # End your test code for IdMap


class UncompressedPostings:

    @staticmethod
    def encode(postings_list):
        """Encodes postings_list into a stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray representing integers in the postings_list
        """
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """Decodes postings_list from a stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray representing encoded postings list as output by encode
            function

        Returns
        -------
        List[int]
            Decoded list of docIDs from encoded_postings_list
        """

        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()


# # Begin your test code for UncompressedPostings
# x = UncompressedPostings.encode([1, 2, 3])
# print(x)
# print(UncompressedPostings.decode(x))
# # End your test code for UncompressedPostings

# ## 磁盘上的倒排索引
class InvertedIndex:
    """A class that implements efficient reads and writes of an inverted index
    to disk

    Attributes
    ----------
    postings_dict: Dictionary mapping: termID->(start_position_in_index_file,
                                                number_of_postings_in_list,
                                               length_in_bytes_of_postings_list)
        This is a dictionary that maps from termIDs to a 3-tuple of metadata
        that is helpful in reading and writing the postings in the index file
        to/from disk. This mapping is supposed to be kept in memory.
        start_position_in_index_file is the position (in bytes) of the postings
        list in the index file
        number_of_postings_in_list is the number of postings (docIDs) in the
        postings list
        length_in_bytes_of_postings_list is the length of the byte
        encoding of the postings list

    terms: List[int]
        A list of termIDs to remember the order in which terms and their
        postings lists were added to index.

        After Python 3.7 we technically no longer need it because a Python dict
        is an OrderedDict, but since it is a relatively new feature, we still
        maintain backward compatibility with a list to keep track of order of
        insertion.
    """

    def __init__(self, index_name, postings_encoding=None, directory=''):
        """
        Parameters
        ----------
        index_name (str): Name used to store files related to the index
        postings_encoding: A class implementing static methods for encoding and
            decoding lists of integers. Default is None, which gets replaced
            with UncompressedPostings
        directory (str): Directory where the index files will be stored
        """
        self.index_file_path = os.path.join(directory, index_name + '.index')
        self.metadata_file_path = os.path.join(directory, index_name + '.dict')

        if postings_encoding is None:
            self.postings_encoding = UncompressedPostings
        else:
            self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []
        # Need to keep track of the order in which the terms were inserted. Would be unnecessary from Python 3.7 onwards

    def __enter__(self):
        """Opens the index_file and loads metadata upon entering the context"""
        # Open the index file
        self.index_file = open(self.index_file_path, 'rb+')

        # Load the postings dict and terms from the metadata file
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms = pkl.load(f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Closes the index_file and saves metadata upon exiting the context"""
        # Close the index file
        self.index_file.close()

        # Write the postings dict and terms to the metadata file
        with open(self.metadata_file_path, 'wb') as f:
            pkl.dump([self.postings_dict, self.terms], f)


# 索引

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): For mapping terms to termIDs
    doc_id_map(IdMap): For mapping relative paths of documents (eg
        0/3dradiology.stanford.edu_) to docIDs
    data_dir(str): Path to data
    output_dir(str): Path to output index files
    index_name(str): Name assigned to index
    postings_encoding: Encoding used for storing the postings.
        The default (None) implies UncompressedPostings
    """

    def __init__(self, data_dir, output_dir, index_name="BSBI",
                 postings_encoding=None):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Stores names of intermediate indices
        self.intermediate_indices = []

    def save(self):
        """Dumps doc_id_map and term_id_map into output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pkl.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pkl.dump(self.doc_id_map, f)

    def load(self):
        """Loads doc_id_map and term_id_map from output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pkl.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pkl.load(f)

    def index(self):
        """Base indexing code

        This function loops through the data directories,
        calls parse_block to parse the documents
        calls invert_write, which inverts each block and writes to a new index
        then saves the id maps and calls merge on the intermediate indices
        """
        for block_dir_relative in sorted(next(os.walk(self.data_dir))[1]):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, directory=self.output_dir,
                                     postings_encoding=
                                     self.postings_encoding) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
        self.save()
        with InvertedIndexWriter(self.index_name, directory=self.output_dir,
                                 postings_encoding=
                                 self.postings_encoding) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexIterator(index_id,
                                          directory=self.output_dir,
                                          postings_encoding=
                                          self.postings_encoding))
                    for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

    def parse_block(self, block_dir_relative):
        """Parses a tokenized text file into termID-docID pairs

        Parameters
        ----------
        block_dir_relative : str
            Relative Path to the directory that contains the files for the block

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block

        Should use self.term_id_map and self.doc_id_map to get termIDs and docIDs.
        These persist across calls to parse_block
        """
        # Begin your code
        result = []
        path = os.path.join(self.data_dir, block_dir_relative)
        if os.path.exists(path) and os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as fr:
                        # !!!一行一行读，才能去掉换行符，分词标准根据空格
                        # text = fr.read()
                        # # text = re.sub(r"[{}]+".format(punctuation), " ", text)
                        # text = text.lower()
                        # text.strip()
                        # text.replace("\n\r", " ")
                        # # words = nltk.word_tokenize(text)
                        # words = text.split(" ")
                        # # words = text.split("\n| ")
                        # words = list(set(words))
                        for line in fr.readlines():
                            line = line.strip()
                            line = line.lower()
                            words = line.split(" ")
                            words = list(set(words))
                            for word in words:
                                term_Id = self.term_id_map[word]
                                # !!!首先让文档处理在文件打开过程中处理，不然就需要重复的打开关闭
                                # !!!这个docId，是不是因为我没有把block_dir_relative加上，导致缺少了很多 => 是的！
                                filename = str(block_dir_relative) + '/' + file
                                doc_Id = self.doc_id_map[filename]
                                # !!!可以直接将它写成tuple，添加到result中，不需要先放[]再转类型
                                result.append((term_Id, doc_Id))
                    # print(result)
        result = list(set(result))
        return result
        # End your code

    def invert_write(self, td_pairs, index):
        """Inverts td_pairs into postings_lists and writes them to the given index

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index on disk corresponding to the block
        """
        # Begin your code
        td_pairs = sorted(td_pairs)
        last_term = td_pairs[0][0]
        last_postings = [td_pairs[0][1]]
        for t in range(1, len(td_pairs)):
            if td_pairs[t][0] != last_term:
                last_postings.sort()
                index.append(last_term, last_postings)
                last_term = td_pairs[t][0]
                last_postings = [td_pairs[t][1]]
            else:
                last_postings.append(td_pairs[t][1])
        last_postings.sort()
        index.append(last_term, last_postings)
        # inverted_index_dict = {}
        # for t in td_pairs:
        #     if t[0] not in list(inverted_index_dict.keys()):
        #
        #         inverted_index_dict[t[0]] = [t[1]]
        #     else:
        #         inverted_index_dict[t[0]].append(t[1])
        # for t in inverted_index_dict.keys():
        #     index.append(t, inverted_index_dict[t])
        # End your code

    def merge(self, indices, merged_index):
        """Merges multiple inverted indices into a single index

        Parameters
        ----------
        indices: List[InvertedIndexIterator]
            A list of InvertedIndexIterator objects, each representing an
            iterable inverted index for a block
        merged_index: InvertedIndexWriter
            An instance of InvertedIndexWriter object into which each merged
            postings list is written out one at a time
        """
        # Begin your code
        # 先构造出要合并的结构，即每一个Iterator中出现的相同的term需要进行合并（docId，term）

        # idx = [0] * (len(indices) + 1)
        # # 设置一个哨兵
        # idx[-1] = -1
        # docId = [[]] * len(indices)
        # print(len(indices))
        # while True:
        #     for i in range(0, len(indices)):
        #         term, doc = next(indices[i])
        #         if term != -1:
        #             idx[i] = term
        #             docId[i] = doc
        #         else:
        #             idx[i] = -1
        #     if set(idx) == {-1}:
        #         break
        #     print(idx)
        #     min_idx = sorted(list(set(idx)))[1]
        #     to_merge_idx = [t for t, v in enumerate(idx) if v == min_idx]
        #     to_merge_docs = []
        #     for k in to_merge_idx:
        #         to_merge_docs.append(docId[k])
        #         docId[k] = []
        #     merged_index.append(min_idx, list(heapq.merge(*to_merge_docs)))
        last_termId = -1
        last_postings = []
        for termId, postings in heapq.merge(*indices, key=lambda x: x[0]):
            if termId != last_termId:
                if last_termId != -1:
                    merged_index.append(last_termId, last_postings)
                last_termId = termId
                last_postings = postings
            else:
                idx1 = 0
                idx2 = 0
                merge_postings = []
                while idx1 < len(last_postings) and idx2 < len(postings):
                    if last_postings[idx1] < postings[idx2]:
                        merge_postings.append(last_postings[idx1])
                        idx1 = idx1 + 1
                    elif last_postings[idx1] > postings[idx2]:
                        merge_postings.append(postings[idx2])
                        idx2 = idx2 + 1
                    else:
                        merge_postings.append(last_postings[idx1])
                        idx1 = idx1 + 1
                        idx2 = idx2 + 1
                if idx1 < len(last_postings):
                    merge_postings.extend(last_postings[idx1:])
                if idx2 < len(postings):
                    merge_postings.extend(postings[idx2:])
                last_postings = merge_postings
                del merge_postings

        if last_termId != -1:
            merged_index.append(last_termId, last_postings)
        # End your code

    def retrieve(self, query):
        """Retrieves the documents corresponding to the conjunctive query

        Parameters
        ----------
        query: str
            Space separated list of query tokens

        Result
        ------
        List[str]
            Sorted list of documents which contains each of the query tokens.
            Should be empty if no documents are found.

        Should NOT throw errors for terms not in corpus
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Begin your code
        tokens = query.strip().split(' ')
        # !!!指定postings_encoding
        with InvertedIndexMapper(self.index_name, directory=self.output_dir,
                                 postings_encoding=self.postings_encoding) as mapper:
            docId = mapper[self.term_id_map[tokens[0]]]
            for t in range(1, len(tokens)):
                list2 = mapper[self.term_id_map[tokens[t]]]
                docId = sorted_intersect(docId, list2)
        doc = []
        print("postings Id: ", end="")
        # print(docId)
        for r in docId:
            if self.doc_id_map[r] != '':
                doc.append(self.doc_id_map[r])
        return doc
        # End your code


class InvertedIndexWriter(InvertedIndex):
    """"""

    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list):
        """Appends the term and postings_list to end of the index file.

        This function does three things,
        1. Encodes the postings_list using self.postings_encoding
        2. Stores metadata in the form of self.terms and self.postings_dict
           Note that self.postings_dict maps termID to a 3 tuple of
           (start_position_in_index_file,
           number_of_postings_in_list,
           length_in_bytes_of_postings_list)
        3. Appends the bytestream to the index file on disk

        Hint: You might find it helpful to read the Python I/O docs
        (https://docs.python.org/3/tutorial/inputoutput.html) for
        information about appending to the end of a file.

        Parameters
        ----------
        term:
            term or termID is the unique identifier for the term
        postings_list: List[Int]
            List of docIDs where the term appears
        """
        # Begin your code
        # !!!需要考虑重复插入的问题吗？
        # step 1.
        # print(postings_list)
        bytes_of_postings = self.postings_encoding.encode(postings_list)
        # step 2.
        # 跳出循环依然没有找到termId的插入位置的话则插入term，否则position_in_index即其所在位置
        if len(self.terms) == 0:
            self.terms.append(term)
            position_in_index = 0
        else:
            # 获取上一个加入到index的词项ID
            post = self.terms[-1]
            self.terms.append(term)
            # 在index的位置为上一个词项的倒排记录表的开始位置 + 它的倒排记录表长度
            # position_in_index = self.index_file.tell()
            position_in_index = self.postings_dict[post][2] + self.postings_dict[post][0]
        number_of_postings = len(postings_list)
        length_of_bytes = len(bytes_of_postings)
        self.postings_dict[term] = (position_in_index, number_of_postings, length_of_bytes)
        # step 3.
        self.index_file.write(bytes_of_postings)
        # End your code


class InvertedIndexIterator(InvertedIndex):
    """"""

    def __enter__(self):
        """Adds an initialization_hook to the __enter__ function of super class
        """
        super().__enter__()
        self._initialization_hook()
        return self

    def _initialization_hook(self):
        """Use this function to initialize the iterator
        """
        # Begin your code
        self.idx = 0
        # !!!这里需要打开index_file吗？
        # End your code

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next (term, postings_list) pair in the index.

        Note: This function should only read a small amount of data from the
        index file. In particular, you should not try to maintain the full
        index file in memory.
        """
        # Begin your code
        if self.idx < len(self.terms):
            term = self.terms[self.idx]
            # !!!从index_file里读出来，这个没有完成！
            postings_start, _, postings_len = self.postings_dict[term]
            if self.index_file.tell() != postings_start:
                self.index_file.seek(postings_start)
            postings_list_encode = self.index_file.read(postings_len)
            postings_list = self.postings_encoding.decode(postings_list_encode)
            self.idx = self.idx + 1
            return term, postings_list
        else:
            # return -1, []
            raise StopIteration
        # End your code

    def delete_from_disk(self):
        """Marks the index for deletion upon exit. Useful for temporary indices
        """
        self.delete_upon_exit = True

    def __exit__(self, exception_type, exception_value, traceback):
        """Delete the index file upon exiting the context along with the
        functions of the super class __exit__ function"""
        self.index_file.close()
        if hasattr(self, 'delete_upon_exit') and self.delete_upon_exit:
            os.remove(self.index_file_path)
            os.remove(self.metadata_file_path)
        else:
            with open(self.metadata_file_path, 'wb') as f:
                pkl.dump([self.postings_dict, self.terms], f)


# # Begin your code for parse_block
# BSBI_instance = BSBIIndex(data_dir=toy_dir, output_dir = 'tmp/', index_name = 'toy')
# print(BSBI_instance.parse_block('0'))
# print(BSBI_instance.term_id_map.str_to_id)
# print(BSBI_instance.doc_id_map.str_to_id)
# # End your code for parse_block


# # Begin your code for invert_write
# with InvertedIndexWriter('test', directory='tmp/') as index:
#     BSBI_instance = BSBIIndex(data_dir=toy_dir, output_dir='tmp/', index_name='toy')
#     td_pairs = BSBI_instance.parse_block('0')
#     BSBI_instance.invert_write(td_pairs, index)
#     print(index.terms)
#     print(index.postings_dict)

# BSBI_instance = BSBIIndex(data_dir=toy_dir, output_dir='tmp/', index_name='toy')
# td_pairs = [(1, 2), (2, 3), (1, 4), (1, 3), (2, 5), (2, 4)]
# with InvertedIndexWriter(BSBI_instance.index_name, BSBI_instance.postings_encoding, BSBI_instance.output_dir) as index:
#     BSBI_instance.invert_write(td_pairs, index)
#     index.index_file.seek(0)
#     assert index.terms == [1, 2], "terms sequence incorrect"
#     assert index.postings_dict == {1: (0, 3, len(UncompressedPostings.encode([2, 3, 4]))),
#                                    2: (len(UncompressedPostings.encode([2, 3, 4])), 3,
#                                        len(UncompressedPostings.encode([3, 4, 5])))}, \
#         "postings_dict incorrect %s".format(index.postings_dict)
#     assert UncompressedPostings.decode(index.index_file.read()) == [2, 3, 4, 3, 4, 5], \
#         "postings on disk incorrect %s".format(index.index_file)
# # End your code for invert_write


# # Begin your code for merge
# with InvertedIndexWriter('test', directory='tmp/') as index:
#     BSBI_instance = BSBIIndex(data_dir=toy_dir, output_dir = 'tmp/', index_name = 'toy')
#     td_pairs= BSBI_instance.parse_block('0')
#     print(td_pairs)
#     BSBI_instance.invert_write(td_pairs, index)
# with InvertedIndexIterator('test', directory='tmp/') as index_iter:
#     for line in index_iter:
#         print(line)
# # End your code for merge


# # Begin your code for index building
# BSBI_instance = BSBIIndex(data_dir=toy_dir, output_dir='toy_output_dir', )
# BSBI_instance.index()
# print(BSBI_instance.term_id_map.str_to_id)
# print(BSBI_instance.term_id_map.id_to_str)
# print(BSBI_instance.doc_id_map.str_to_id)


# BSBI_instance = BSBIIndex(data_dir='pal-data', output_dir='output_dir', )
# BSBI_instance.index()


# BSBI_instance = BSBIIndex(data_dir='pa1-data', output_dir='output_dir', ) BSBI_instance.intermediate_indices = [
# 'index_'+str(i) for i in range(10)] with InvertedIndexWriter(BSBI_instance.index_name,
# directory=BSBI_instance.output_dir, postings_encoding=BSBI_instance.postings_encoding) as merged_index: with
# contextlib.ExitStack() as stack: indices = [stack.enter_context(InvertedIndexIterator(index_id,
# directory=BSBI_instance.output_dir, postings_encoding=BSBI_instance.postings_encoding)) for index_id in
# BSBI_instance.intermediate_indices] BSBI_instance.merge(indices, merged_index)

# # End your code for index building

class InvertedIndexMapper(InvertedIndex):
    def __getitem__(self, key):
        return self._get_postings_list(key)

    def _get_postings_list(self, term):
        """Gets a postings list (of docIds) for `term`.

        This function should not iterate through the index file.
        I.e., it should only have to read the bytes from the index file
        corresponding to the postings list for the requested term.
        """
        # Begin your code
        if term in self.postings_dict.keys():
            posting_location, posting_number, posting_length = self.postings_dict[term]
            # print(posting_location, posting_number, posting_length)
            self.index_file.seek(posting_location)
            postings = self.index_file.read(posting_length)
            postings_list = self.postings_encoding.decode(postings)
            # print(postings_list)
            return postings_list
        else:
            return []
        # End your code


# # Begin your code for InvertedIndexMapper
# BSBI_instance = BSBIIndex(data_dir=toy_dir, output_dir='toy_output_dir', )
# BSBI_instance.load()
# with InvertedIndexMapper('BSBI', directory='toy_output_dir') as mapper:
#     print(mapper.postings_dict)
#     for key in mapper.postings_dict.keys():
#         print(BSBI_instance.term_id_map[key])
#     print(mapper[BSBI_instance.term_id_map['bye']])
#     for idx in mapper[BSBI_instance.term_id_map['bye']]:
#         print(BSBI_instance.doc_id_map[idx])
#
# BSBIIndex_instance = BSBIIndex(data_dir='toy-data', output_dir='toy_output_dir')
# BSBIIndex_instance.load()
# with InvertedIndexMapper('BSBI', directory='toy_output_dir') as mapper:
#     print(mapper.postings_dict.keys())
#     print(mapper[BSBIIndex_instance.term_id_map['thank']])
# # End your code for InvertedIndexMapper


def sorted_intersect(list1, list2):
    """Intersects two (ascending) sorted lists and returns the sorted result

    Parameters
    ----------
    list1: List[Comparable]
    list2: List[Comparable]
        Sorted lists to be intersected

    Returns
    -------
    List[Comparable]
        Sorted intersection
    """
    # Begin your code
    idx1 = 0
    idx2 = 0
    result = []
    list1 = sorted(list1)
    list2 = sorted(list2)
    while idx1 < len(list1) and idx2 < len(list2):
        if list1[idx1] < list2[idx2]:
            idx1 = idx1 + 1
        elif list1[idx1] > list2[idx2]:
            idx2 = idx2 + 1
        else:
            result.append(list1[idx1])
            idx1 = idx1 + 1
            idx2 = idx2 + 1
    return result
    # End your code


# # Begin your code for Index
# BSBI_instance = BSBIIndex(data_dir=toy_dir, output_dir = 'toy_output_dir', )
# BSBI_instance.load()
# with InvertedIndexMapper('BSBI', directory='toy_output_dir') as mapper:
#     list1 = mapper[BSBI_instance.term_id_map['you']]
#     list2 = mapper[BSBI_instance.term_id_map['hi']]
#     print(list1)
#     print(list2)
#     print(sorted_intersect(list1, list2))
#
# res = BSBI_instance.retrieve('hi you')
# my_results = [os.path.normpath(path) for path in BSBI_instance.retrieve('hi you')]
# print(res)
# print(my_results)

BSBI_instance = BSBIIndex(data_dir='pal-data', output_dir='output_dir', )
# BSBI_instance.load()
# res = BSBI_instance.retrieve('we are')
# with open('dev_output/1.out', 'r') as fw:
#     # for r in res:
#     #     fw.write(r)
#     #     fw.write('\n')
#     reference_results = [x.strip() for x in fw.readlines()]
#     print(len(set(res)))
#     print(len(set(reference_results)))
#     print(set(res) - set(reference_results))
#     print(set(reference_results) - set(res))


# for l in res:
#     print(l)

#
# try:
#     os.mkdir('my_output')
# except FileExistsError:
#     pass
#
# for i in range(1, 9):
#     with open('dev_queries/query.' + str(i)) as q:
#         query = q.read()
#         my_results = [os.path.normpath(path) for path in BSBI_instance.retrieve(query)]
#         with open('my_output/' + str(i) + '.out', 'w') as mo:
#             mo.writelines(my_results)
print("--------------------No Compressed--------------------")
for i in range(1, 9):
    with open('dev_queries/query.' + str(i)) as q:
        query = q.read().strip()
        my_results = [os.path.normpath(path) for path in BSBI_instance.retrieve(query)]
    with open('dev_output/' + str(i) + '.out') as o:
        reference_results = [os.path.normpath(x.strip()) for x in o.readlines()]
        # print(len(my_results))
        # print(len(reference_results))
        # print(set(my_results) - set(reference_results))
        # print(set(reference_results) - set(my_results))
        assert set(my_results) == set(reference_results), "Results DO NOT match for query: " + query.strip()
        # assert my_results == reference_results, "Results DO NOT match for query: " + query.strip()
    print("Results match for query:", query.strip())

# # End your code for Index

class CompressedPostings:
    # If you need any extra helper methods you can add them here
    # Begin your code
    @staticmethod
    def VBEncode(num):
        result = []
        while int(num / 128) != 0:
            result.append(num % 128)
            num = int(num / 128)
        result.append(num % 128)
        result[0] = result[0] + 128
        return result[::-1]

    @staticmethod
    def VBDecode(bytestream):
        result = []
        last_num_list = []
        for byte in bytestream:
            if byte // 128 > 0:
                num = 0
                if last_num_list:
                    for i in range(0, len(last_num_list)):
                        num = num + last_num_list[i] * pow(2, 7 * (len(last_num_list) - i))
                num = num + byte - 128
                result.append(num)
                last_num_list = []
            else:
                last_num_list.append(byte)
        return result

    # End your code

    @staticmethod
    def encode(postings_list):
        """Encodes `postings_list` using gap encoding with variable byte
        encoding for each gap

        Parameters
        ----------
        postings_list: List[int]
            The postings list to be encoded

        Returns
        -------
        bytes:
            Bytes reprsentation of the compressed postings list
            (as produced by `array.tobytes` function)
        """
        # Begin your code
        # gap encoding
        first = postings_list[0]
        gap_posting_list = CompressedPostings.VBEncode(first)
        for i in range(1, len(postings_list)):
            gap = postings_list[i] - postings_list[i - 1]
            vbgap = CompressedPostings.VBEncode(gap)
            gap_posting_list.extend(vbgap)
        return array.array('B', gap_posting_list).tobytes()
        # End your code

    @staticmethod
    def decode(encoded_postings_list):
        """Decodes a byte representation of compressed postings list

        Parameters
        ----------
        encoded_postings_list: bytes
            Bytes representation as produced by `CompressedPostings.encode`

        Returns
        -------
        List[int]
            Decoded postings list (each posting is a docIds)
        """
        # Begin your code
        decoded_postings_list = array.array('B')
        decoded_postings_list.frombytes(encoded_postings_list)
        decoded_postings_list = decoded_postings_list.tolist()
        decoded_postings_list = CompressedPostings.VBDecode(decoded_postings_list)
        first = decoded_postings_list[0]
        postings_list = [first]
        for i in range(1, len(decoded_postings_list)):
            postings_list.append(postings_list[-1] + decoded_postings_list[i])
        return postings_list
        # End your code


# # Begin your code for CompressedPostings
# postings_list = [824, 829, 215406]
# encoded_postings_list = CompressedPostings.encode(postings_list)
# print(encoded_postings_list)
# res = CompressedPostings.decode(encoded_postings_list)
# print(res)
# def test_encode_decode(l):
#     e = CompressedPostings.encode(l)
#     d = CompressedPostings.decode(e)
#     assert d == l
#     print(l, e)
#
#
# test_encode_decode([12, 19, 34])
# # End your code for CompressedPostings

# # Begin your code for CompressedIndex building
#
# try:
#     os.mkdir('output_dir_compressed')
# except FileExistsError:
#     pass
#
# BSBI_instance_compressed = BSBIIndex(data_dir='pal-data', output_dir = 'output_dir_compressed', postings_encoding=CompressedPostings)
# BSBI_instance_compressed.index()
#
BSBI_instance_compressed = BSBIIndex(data_dir='pal-data', output_dir='output_dir_compressed', postings_encoding=CompressedPostings)
# BSBI_instance_compressed.retrieve('boolean retrieval')
print("--------------------VB Compressed--------------------")
for i in range(1, 9):
    with open('dev_queries/query.' + str(i)) as q:
        query = q.read().strip()
        my_results = [os.path.normpath(path) for path in BSBI_instance_compressed.retrieve(query)]
    with open('dev_output/' + str(i) + '.out') as o:
        reference_results = [os.path.normpath(x.strip()) for x in o.readlines()]
        # print(len(my_results))
        # print(len(reference_results))
        # print(set(my_results) - set(reference_results))
        # print(set(reference_results) - set(my_results))
        assert set(my_results) == set(reference_results), "Results DO NOT match for query: " + query.strip()
    print("Results match for query:", query.strip())
# # End your code for CompressedIndex building


class ECCompressedPostings:
    # If you need any extra helper methods you can add them here
    # Begin your code
    @staticmethod
    def gammaEncode(gap_postings_list):
        intermediate_str = ''
        result = []
        if gap_postings_list[0] == 0:
            intermediate_str = '0'
            idx = 1
        else:
            idx = 0
        for g in gap_postings_list[idx:]:
            g = g + 1
            binary_str = bin(g).replace("0b", "")
            one_num = len(binary_str) - 1
            binary_res = '1' * one_num + '0' + binary_str[1:]
            intermediate_str = intermediate_str + binary_res
        if len(intermediate_str) % 8 != 0:
            fill_num = 8 - len(intermediate_str) % 8
            intermediate_str = intermediate_str + '1' * fill_num
        t = 0
        while t * 8 < len(intermediate_str):
            sub_binary_str = intermediate_str[t * 8:t * 8 + 8]
            decimal_res = 0
            for i in range(0, 8):
                decimal_res = decimal_res + int(sub_binary_str[i]) * pow(2, 7 - i)
            result.append(decimal_res)
            t = t + 1
        return result

    @staticmethod
    def gammaDecode(bytestream):
        result = []
        intermediate_str = ''
        # if bytestream[0] == 0:
        #     result.append(0)
        # else:
        # intermediate_str = "{0:b}".format(bytestream[0])
        for t in range(0, len(bytestream)):
            byte = "{:08b}".format(bytestream[t])
            intermediate_str += byte
        count = 0
        i = 0
        # print(intermediate_str)
        while i < len(intermediate_str):
            if intermediate_str[i] == '1':
                count += 1
                i = i + 1
            elif intermediate_str[i] == '0' and count != 0:
                num_str = '1' + intermediate_str[i + 1: i + 1 + count]
                num = 0
                for j in range(0, len(num_str)):
                    num += int(num_str[j]) * pow(2, len(num_str) - 1 - j)
                result.append(num - 1)
                i = i + 1 + count
                count = 0
            elif intermediate_str[i] == '0' and count == 0:
                result.append(0)
                i = i + 1
        return result

    @staticmethod
    def encode(postings_list):
        """Encodes `postings_list` using gap encoding with variable byte
        encoding for each gap

        Parameters
        ----------
        postings_list: List[int]
            The postings list to be encoded

        Returns
        -------
        bytes:
            Bytes reprsentation of the compressed postings list
            (as produced by `array.tobytes` function)
        """
        # Begin your code
        # gap encoding
        # print("encode, postings: ", end="")
        # print(postings_list)
        first = postings_list[0]
        gap_posting_list = [first]
        for i in range(1, len(postings_list)):
            gap = postings_list[i] - postings_list[i - 1]
            gap_posting_list.append(gap)
        # print(gap_posting_list)
        gap_posting_list = ECCompressedPostings.gammaEncode(gap_posting_list)
        # print("encoded: ", end="")
        # print(gap_posting_list)
        return array.array('B', gap_posting_list).tobytes()
        # End your code

    @staticmethod
    def decode(encoded_postings_list):
        """Decodes a byte representation of compressed postings list

        Parameters
        ----------
        encoded_postings_list: bytes
            Bytes representation as produced by `CompressedPostings.encode`

        Returns
        -------
        List[int]
            Decoded postings list (each posting is a docIds)
        """
        # Begin your code
        # print(encoded_postings_list)
        decoded_postings_list = array.array('B')
        decoded_postings_list.frombytes(encoded_postings_list)
        decoded_postings_list = decoded_postings_list.tolist()
        # print(decoded_postings_list)
        decoded_postings_list = ECCompressedPostings.gammaDecode(decoded_postings_list)
        # print(decoded_postings_list)
        first = decoded_postings_list[0]
        postings_list = [first]
        for i in range(1, len(decoded_postings_list)):
            postings_list.append(postings_list[i - 1] + decoded_postings_list[i])
        # print("decode, postings: ", end="")
        # print(postings_list)
        # postings_list = postings_list[::-1]
        return postings_list
        # End your code


# Begin your code for ECCompressedPostingsIndex building
# try:
#     os.mkdir('output_dir_eccompressed')
# except FileExistsError:
#     pass

# BSBI_instance_eccompressed = BSBIIndex(data_dir='pal-data', output_dir='output_dir_eccompressed', postings_encoding=ECCompressedPostings)
# BSBI_instance_eccompressed.index()
# End your code for ECCompressedPostingsIndex building

BSBI_instance_eccompressed = BSBIIndex(data_dir='pal-data', output_dir='output_dir_eccompressed', postings_encoding=ECCompressedPostings)
# BSBI_instance_eccompressed.index()
# res = BSBI_instance_eccompressed.retrieve('boolean')
# print(len(res))
# print(res)
# res = BSBI_instance_eccompressed.retrieve('retrieval')
# print(len(res))
# print(res)
print("--------------------Gamma Compressed--------------------")
for i in range(1, 9):
    with open('dev_queries/query.' + str(i)) as q:
        query = q.read().strip()
        res = BSBI_instance_eccompressed.retrieve(query)
        # print(res)
        my_results = [os.path.normpath(path) for path in res]
    with open('dev_output/' + str(i) + '.out') as o:
        reference_results = [os.path.normpath(x.strip()) for x in o.readlines()]
        # print(len(my_results))
        # print(len(reference_results))
        # print(set(my_results) - set(reference_results))
        # print(set(reference_results) - set(my_results))
        assert set(my_results) == set(reference_results), "Results DO NOT match for query: " + query.strip()
    print("Results match for query:", query.strip())

# try:
#     os.mkdir('toy_output_dir_eccompressed')
# except FileExistsError:
#     pass

# BSBI_instance = BSBIIndex(data_dir=toy_dir, output_dir='toy_output_dir_eccompressed', postings_encoding=ECCompressedPostings)
# BSBI_instance.index()
# res = BSBI_instance.retrieve('bye')
# print(res)
#
# def test_encode_decode(l):
#     e = ECCompressedPostings.encode(l)
#     d = ECCompressedPostings.decode(e)
#     assert d == l
#     print(l, e)
#
#
# test_encode_decode([127])
