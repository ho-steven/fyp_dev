import streamlit as st  # pip install streamlit


def callus():

    st.header(":mailbox: Contact us")
    st.write("All fields are required. Your email address will not be published.")


    contact_form = """
        <form action="https://formsubmit.co/walan566@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Name" required>
        <input type="email" name="email" placeholder="Email" required>
        <textarea name="message" placeholder="Message"></textarea>
        <button type="submit">Send</button>
        </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)



    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css("style/style.css.txt")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.info("Future work")
    with st.expander("See Details  👁 "):
        st.subheader("Upcoming currencies")
        st.write(1, "  Terra (LUNA)")
        st.write(2, "  Solana (SOL)")
        st.write(3, "  XRP (XRP)")
        st.write(4, "  Avalanche (AVAX)")
        st.write(5, "  Ethereum Classic (ETC)")

