# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from typing import List
import string
import random
from data_utils import *
from rnn import *
import torch
import codecs
from tqdm import tqdm
import string

#Set GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load vocabulary files
input_lang = torch.load('data-bin/fra.data')
output_lang = torch.load('data-bin/eng.data')

#Create and empty RNN model
encoder = EncoderRNN(input_size=input_lang.n_words, device=device)
attn_decoder = AttnDecoderRNN(output_size=output_lang.n_words, device=device)

#Load the saved model weights into the RNN model
encoder.load_state_dict(torch.load('model/encoder'))
attn_decoder.load_state_dict(torch.load('model/decoder'))

#Return the decoder output given input sentence 
#Additionally, the previous predicted word and previous decoder state can also be given as input
def translate_single_word(encoder, decoder, sentence, decoder_input=None, decoder_hidden=None, max_length=MAX_LENGTH, device=device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        if decoder_input==None:
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        else:
            decoder_input = torch.tensor([[output_lang.word2index[decoder_input]]], device=device) 
        
        if decoder_hidden == None:        
            decoder_hidden = encoder_hidden
        
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        return decoder_output.data, decoder_hidden

#########################################################################################
#####Modify this function to use beam search to predict instead of greedy prediction#####
#########################################################################################
def beam_search(encoder,decoder,input_sentence,beam_size=1,max_length=MAX_LENGTH):
    decoded_beam: List[List[tuple]] = []
    decoded_output: List[List[tuple]] = []
    
    # Predict the first word
    decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, decoder_input=None, decoder_hidden=None)
    
    # Get the probability of all possible words -> (1, n) tensor
    decoder_output_probs = decoder_output.data

    # Get the decoder output size -> n
    decoder_output_size = decoder_output_probs.size()[1]

    # Sort the probabilities in descending order and get the indices -> (beam_size,) tensor
    top_indices = torch.argsort(decoder_output_probs, descending=True).squeeze()[:beam_size]

    # Convert each predicted idx into the word and add it to the decoded beam
    for idx in top_indices:
        idx = idx.item()
        first_word = output_lang.index2word[idx]  # Get the word
        first_word_prob = decoder_output_probs[0][idx].item()  # Get the probability for that word

        # If it is an EOS token, add empty line to decoded output
        if idx == EOS_token:
            decoded_output.append([("", first_word_prob, decoder_hidden)])
        # Otherwise, add it to the beam for searching
        else:
            decoded_beam.append([(first_word, first_word_prob, decoder_hidden)])

    # Loop until the maximum length
    for _ in range(max_length):

        tmp_decoded_beam = []
        hidden_states = []
        decoder_outputs = []

        # Run prediction for each word in the decoded beam
        for beam in decoded_beam:
            # Get the previous decoded output, its probability, and the previous decoder hidden state
            previous_decoded_output, previous_decoded_output_prob, decoder_hidden = beam[-1]

            # Predict the next word given the previous prediction and the previous decoder hidden state
            decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, previous_decoded_output, decoder_hidden)

            # Store the hidden state
            hidden_states.append(decoder_hidden)

            # Store probability for combination of words
            decoder_outputs.append(previous_decoded_output_prob * decoder_output.data)
        
        # Sort the probabilities in descending order and get the top beam_size indexes -> (beam_size,) tensor
        concatenated_probs = torch.cat(decoder_outputs, axis=1)
        top_indices = torch.argsort(concatenated_probs, descending=True).squeeze()[:len(decoded_beam)]

        for idx in top_indices:

            # Get the relevant beam index and the true index for lookups
            idx = idx.item()
            beam_index = idx // decoder_output_size
            index = idx - (decoder_output_size * beam_index)

            # Get the relevant copies
            beam = decoded_beam[beam_index].copy()
            decoder_hidden = hidden_states[beam_index]

            # Add it to the final list if it is the EOS token
            if index == EOS_token:
                decoded_output.append(beam)
            # Get the decoded word, its probability and add it to the next beam search
            else:
                decoded_word = output_lang.index2word[index]
                decoded_word_prob = concatenated_probs[0][idx].item()
                beam.append((decoded_word, decoded_word_prob, decoder_hidden))
                tmp_decoded_beam.append(beam)

        decoded_beam = tmp_decoded_beam
        if not decoded_beam:
            break
        

    # In case any of the sentences did not get an EOS token within the maximum length
    if decoded_beam:
        decoded_output.extend(decoded_beam)

    if len(decoded_output) != beam_size:
        raise Exception(f"Unexpected Error: Decoded Output Length != Beam Size - {len(decoded_output)} != {beam_size}")

    # Get the output with the maximum probability
    sorted_decoded_output = sorted(decoded_output, key=lambda x: x[-1][1], reverse=True)
    most_likely_output = [item[0] for item in sorted_decoded_output[0]]
    
    #Convert list of predicted words to a sentence and detokenize 
    output_translation = " ".join(most_likely_output)
    
    return output_translation


with open('data/test.eng', 'r') as f:
    target_sentences = f.read().strip().split('\n')

with open('data/test.fra', 'r') as f:
    source_sentences = f.read().strip().split('\n')


target = codecs.open(f'outputs/test_beam_6.out', 'w', encoding='utf-8')
beam_size = 6
for i,source_sentence in enumerate(source_sentences):

    target_sentence = normalizeString(target_sentences[i])
    input_sentence = normalizeString(source_sentence)
    hypothesis = beam_search(encoder, attn_decoder, input_sentence, beam_size=beam_size)
    
    print("S-"+str(i)+": "+input_sentence)
    print("T-"+str(i)+": "+target_sentence)
    print("H-"+str(i)+": "+hypothesis)
    print()
    target.write(hypothesis+'\n')
target.close()
