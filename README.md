# Machine Translation for Subtitles
Machine translation using sequence-to-sequence modeling technique using RNN to translate from Spanish to English language.

## Installation
1. To run this program, please install tensorflow 1.0.1 either CPU or GPU version. 
2. Once installed download the parallel corpora from OPUS(`http://opus.lingfil.uu.se/OpenSubtitles2016.php`) collection of subtitles from `OpenSubtitles.org`
3. Clone the repository into your local machine. `git clone https://github.com/mehulviby/subtitle_translation.git`
4. Save the downloaded parallel corpus inside the `translation/corpus/` folder with the name given

## Running
To train the corpus program type in `python translate.py --data_dir [corpus directory] --train_dir [train directory]`

Parameters can be modified to get better results

1. `--size=256` 										- Size of the each layer being saved will be 256 units
2. `--num_layers=2` 								- Number of RNN layers used to train the model.
3. `--steps_per_checkpoint=50` 			- size per checkpoint to train and save the data

The final train information will be stored in "translation/train" folder.</br>
To decode the corpus program type in `python translate.py --decode --data_dir [corpus directory] --train_dir [train directory]`

The decoded information will be stored in "translation/decode" folder

## Evaluation

Three types of evaluation done:
1. Corpus Evaluation
2. Word-to-word translation for baseline score
3. Machine Translation Evaluation - BLEU score evalutaion

To evaluate the decoded information, get the decoded file `en_decode_output.txt` from translation/decode folder</br>
Transfer this to BLEU/reference folder along with original file as candidate.txt into BLEU/candidate folder.

To calculate BLEU score type `python calculateBleu3.py candidate/candidate.txt reference/`
