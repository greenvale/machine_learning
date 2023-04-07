#include <iostream>
#include <fstream>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <iomanip>

#include "nlp_useful.hpp"

// Apostrophe DEC ASCII code = 39

/****************************************************************************************************************/


int main()
{
    // get sentences
    auto sentences = gv::nlp::get_speech_sentences(
        "./data/starmen.txt", "./data/eng_dictionary.txt", 8, {(char)39});

    // get lexicon
    auto p_lex = gv::nlp::get_lexicon_ptr(gv::flatten_vector(sentences));
    std::vector<std::string> lex(p_lex->begin(), p_lex->end());
    delete p_lex;

    // one-hot encode sentences
    auto oh = gv::nlp::to_onehot(sentences, lex);

    

}