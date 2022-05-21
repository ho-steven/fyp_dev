import streamlit as st

def aus():

    # st.title("Abstract")
    # st.write("")
    # st.write("")
    # st.subheader("In the past years we have seen that price of cryptocurrency rapidly grow due to its great return. "
             # "So, our team explored several cryptocurrencies and we have decided to make a web-app that will display all "
             # "the necessary things (like Basic info of particular crypto, their live pricing, 7 days graph, etc) along with hourly, "
            # "weekly and monthly forecasting of crypto coins for better understanding of the nature of cryptocurrencies.")
    # st.write("---")

    st.subheader("Project Supervisor")
    st.markdown("***")

    col1, col2, col3 = st.columns([3, 1, 9])
    with col1:

        st.image("Images/JurgenRahmel.jpg", use_column_width=True)

    with col3:
        st.write("")
        st.write("")
        st.write("")
        st.title("Dr. Juergen Rahmel")
        st.write("Visiting Lecturer, Department of Computer Science, HKU")
        st.write("")
        st.subheader("Dr. Juergen Rahmel holds a PhD in computer science (Artificial Intelligence, Neural Networks) from University of Kaiserslautern in Germany and an MBA in International Management from University of London.")
        st.subheader("He is an experienced part-time university lecturer on a variety of subjects, including for example Information Security, eCommerce, Enterprise Resource Planning, Securities Transaction Banking, and eFinancial Services.")

#He started his career with Deutsche Bank in Frankfurt, Germany, has worked in various IT management positions of different banks and has transferred for HSBC from Germany to Hong Kong in 2010.
#In parallel, Juergen started his own company where he coaches individuals or groups on business-related IT matters, academic writing and other related subjects.  One endeavour particularly close to his heart is the education of children (and adults - they need it too...) in Internet safety for them to become self-confident digital citizens.

    st.markdown("***")
    st.subheader("Project Team Memebers")
    st.markdown("***")

    col4,col5,col6 = st.columns([9,1,3])
    with col4:
       st.write("")
       st.write("")
       st.write("")
       st.title("Chow Pui Yin Roy")
       st.write("Project Leader")
       st.write("Part-time Msc(CompSc) student in the University of Hong Kong")
       st.write("")
       st.subheader("Lead the team to develop end-to-end roll out plan from ideation, data analysis, build and design on web application development. Outline the scope of the project and methodologies.")

    with col6:
        st.image("Images/man.jpg", use_column_width=True)

    st.markdown("***")

    col7,col8,col9 = st.columns([9,1,3])
    with col7:
       st.write("")
       st.write("")
       st.write("")
       st.title("Ho Kam Tim")
       st.write("Part-time Msc(CompSc) student in the University of Hong Kong")
       st.write("")
       st.subheader("Focus on Web/app development and programming based on group direction and discussion, support the team on technical feasibility study and web design.")

    with col9:
        st.image("Images/man.jpg", use_column_width=True)

    st.markdown("***")

    # col7, ccl8, col9 = st.columns([3, 1, 9])
    # with col7:
        # st.image("Images/mihir.jpg", use_column_width=True)

    # with col9:
        # st.write("")
        # st.write("")
        # st.write("")
        # st.title("MIHIR MATHUR")
        # st.caption("Languages : HTML, CSS, JavaScript, C, C++,  Java, Python")
        # st.write("")
        # st.subheader("I had completed Bachelor's of Computer Science and Engineering from JECRC University" +
                     # " and currently pursuing Post-Graduation Certificate in Cloud Computing for Big Data.")



    col10, col11,col12 = st.columns([9,1,3])
    with col10:
        st.write("")
        st.write("")
        st.write("")
        st.title("Chan Po Yi")
        st.write("Part-time Msc(CompSc) student in the University of Hong Kong ")
        st.write("")
        st.subheader("Focus on literature review in research methodology, data analysis, and project framework design, support the team on identify report flow, visuals and presentation.")
    with col12:
        st.image("Images/girl.jpg", use_column_width=True)

    hide_img_fs = '''
                    <style>
                    button[title="View fullscreen"]{
                        visibility: hidden;}
                    </style>
                    '''

    st.markdown(hide_img_fs, unsafe_allow_html=True)