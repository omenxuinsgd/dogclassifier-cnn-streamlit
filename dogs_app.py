import os
import requests
from PIL import ExifTags
from fastai.learner import load_learner
from fastai.vision.core import PILImage
import streamlit as st

REPO_DIR = 'https://github.com/willjobs/dog-classifier/raw/main'
MODEL_FILE = 'dogs_online_resnet50_cpu.pkl'

st.set_page_config(
    page_title="Dog Classifier - Will Jobs",
    page_icon="🐶"
)

st.write("# Dog Breed Classifier")
st.markdown('By <a href="https://github.com/omenxuinsgd" target="_blank">Nur Rokhman</a>' , unsafe_allow_html=True)

st.write("Proyek ini mengklasifikasikan foto anjing menggunakan CNN yang disetel dengan baik dari ResNet-50 di fastai.")
with st.beta_expander("🧙 Klik di sini untuk info lebih lanjut tentang model 🔮"):
    st.markdown("""
        <p>Proyek ini menggunakan pembelajaran transfer (transfer learning) untuk membuat CNN yang telah dilatih sebelumnya di ImageNet, menggunakan
        arsitektur ResNet-50. Implementasi dilakukan menggunakan fastai (v2) dan PyTorch.
        Dataset pelatihan didasarkan pada <a href="https://www.akc.org/dog-breeds/" target="_blank">AKC-recognized dog breeds</a>, 
        dengan sekitar 150 gambar per jenis anjing yang diambil dari internet. (Iterasi sebelumnya dari
        proyek ini menggunakan <a href="http://vision.stanford.edu/aditya86/ImageNetDogs/" target="_blank">Stanford dogs dataset</a>, 
        tetapi saya menemukan bahwa gambar dalam kumpulan data itu tidak mewakili gambar "di alam liar", seperti
        model yang sama yang dilatih pada gambar tersebut memberikan akurasi tinggi yang tidak realistis tetapi tidak menggeneralisasi dengan baik.)</p>

        <p>10% dari data disisihkan untuk set pengujian (holdout set), dan 20% dari data
        digunakan untuk set validasi. Gambar diubah ukurannya menjadi kotak 128x128 piksel sebelumnya
        pelatihan menggunakan pemotongan ukuran acak.</p>

        <p>Model terakhir dilatih untuk total 8 zaman: 3 dengan lapisan ResNet dibekukan,
        hanya melatih kepala klasifikasi baru, dan 5 zaman tambahan dengan semua lapisan dicairkan.
        <b>Akurasi set validasi terakhir adalah 76,3%</b>, dan <b>akurasi set/holdout pengujian adalah 75,4%</b>.</p>

        <p>Kode yang digunakan untuk melatih model tersedia di <a href="https://github.com/omenxuinsgd/dogclassifier-cnn-streamlit" target="_blank">https://github.com/omenxuinsgd/dogclassifier-cnn-streamlit</a>.</p>
    """, unsafe_allow_html=True)

file_data = st.file_uploader("Pilih sebuah gambar", type=["jpg", "jpeg", "png"])


def download_file(url):
    with st.spinner('Downloading model...'):
        # from https://stackoverflow.com/a/16696317
        local_filename = url.split('/')[-1]
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk: 
                    f.write(chunk)
        return local_filename

def fix_rotation(file_data):
    # check EXIF data to see if has rotation data from iOS. If so, fix it.
    try:
        image = PILImage.create(file_data)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        exif = dict(image.getexif().items())

        rot = 0
        if exif[orientation] == 3:
            rot = 180
        elif exif[orientation] == 6:
            rot = 270
        elif exif[orientation] == 8:
            rot = 90

        if rot != 0:
            st.write(f"Rotating image {rot} degrees (you're probably on iOS)...")
            image = image.rotate(rot, expand=True)
            # This step is necessary because image.rotate returns a PIL.Image, not PILImage, the fastai derived class.
            image.__class__ = PILImage

    except (AttributeError, KeyError, IndexError):
        pass  # image didn't have EXIF data

    return image


# cache the model so it only gets loaded once
@st.cache(allow_output_mutation=True)
def get_model():
    if not os.path.isfile(MODEL_FILE):
        _ = download_file(f'{REPO_DIR}/models/{MODEL_FILE}')

    learn = load_learner(MODEL_FILE)
    return learn

learn = get_model()

if file_data is not None:
    with st.spinner('Classifying...'):
        # load the image from uploader; fix rotation for iOS devices if necessary
        img = fix_rotation(file_data)
        
        st.write('## Your Image')
        st.image(img, width=200)

        # classify
        pred, pred_idx, probs = learn.predict(img)
        top5_preds = sorted(list(zip(learn.dls.vocab, list(probs.numpy()))), key=lambda x: x[1], reverse=True)[:5]

        # prepare output
        out_text = '<table><tr> <th>Breed</th> <th>Confidence</th> <th>Example</th> </tr>'

        for pred in top5_preds:
            example = REPO_DIR + '/example_dogs/' + pred[0].replace(" ", "").lower() + ".jpg"
            out_text += '<tr>' + \
                            f'<td>{pred[0]}</td>' + \
                            f'<td>{100 * pred[1]:.02f}%</td>' + \
                            f'<td><img src="{example}" height="150" /></td>' + \
                        '</tr>'
        out_text += '</table><br><br>'

        st.write('## Apa yang model pikirkan')
        st.markdown(out_text, unsafe_allow_html=True)

        st.write(f"🤔 Tidak melihat ras anjing Anda? Untuk daftar lengkap ras anjing dalam proyek ini, [klik di sini](https://htmlpreview.github.io/?https://github.com/willjobs/dog-classifier/blob/main/dog_breeds.html).")
