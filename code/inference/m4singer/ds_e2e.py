import torch
# from inference.tts.fs import FastSpeechInfer
# from modules.tts.fs2_orig import FastSpeech2Orig
from inference.m4singer.base_svs_infer import BaseSVSInfer
from utils import load_ckpt
from utils.hparams import hparams
from usr.diff.shallow_diffusion_tts import GaussianDiffusion
from usr.diffsinger_task import DIFF_DECODERS
from modules.fastspeech.pe import PitchExtractor
import utils


class DiffSingerE2EInfer(BaseSVSInfer):
    def build_model(self):
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().to(self.device)
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        with torch.no_grad():
            output = self.model(txt_tokens, spk_embed=spk_id, ref_mels=None, infer=True,
                                pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
                                is_slur=sample['is_slur'])
            mel_out = output['mel_out']  # [B, T,80]
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                f0_pred = self.pe(mel_out)['f0_denorm_pred']  # pe predict from Pred mel
            else:
                f0_pred = output['f0_denorm']
            wav_out = self.run_vocoder(mel_out, f0=f0_pred)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]

if __name__ == '__main__':
    inp = {
        'spk_name': 'Tenor-1',
        'text': 'AP你要相信AP相信我们会像童话故事里AP',
        'notes': 'rest | G#3 | A#3 C4 | D#4 | D#4 F4 | rest | E4 F4 | F4 | D#4 A#3 | A#3 | A#3 | C#4 | B3 C4 | C#4 | B3 C4 | A#3 | G#3 | rest',
        'notes_duration': '0.14 | 0.47 | 0.1905 0.1895 | 0.41 | 0.3005 0.3895 | 0.21 | 0.2391 0.1809 | 0.32 | 0.4105 0.2095 | 0.35 | 0.43 | 0.45 | 0.2309 0.2291 | 0.48 | 0.225 0.195 | 0.29 | 0.71 | 0.14',
        'input_type': 'word',
    } 

    c = {
        'spk_name': 'Tenor-1',
        'text': '你要相信相信我们会像童话故事里',
        'ph_seq': '<AP> n i iao iao x iang x in in <AP> x iang iang x in uo uo m en h uei x iang t ong ong h ua g u u sh i l i <AP>',
        'note_seq': 'rest G#3 G#3 A#3 C4 D#4 D#4 D#4 D#4 F4 rest E4 E4 F4 F4 F4 D#4 A#3 A#3 A#3 A#3 A#3 C#4 C#4 B3 B3 C4 C#4 C#4 B3 B3 C4 A#3 A#3 G#3 G#3 rest',
        'note_dur_seq': '0.14 0.47 0.47 0.1905 0.1895 0.41 0.41 0.3005 0.3005 0.3895 0.21 0.2391 0.2391 0.1809 0.32 0.32 0.4105 0.2095 0.35 0.35 0.43 0.43 0.45 0.45 0.2309 0.2309 0.2291 0.48 0.48 0.225 0.225 0.195 0.29 0.29 0.71 0.71 0.14',
        'is_slur_seq': '0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0',
        'input_type': 'phoneme'
    }
    DiffSingerE2EInfer.example_run(inp)
