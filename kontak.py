import streamlit as st
import streamlit.components.v1 as components

def contact_me():
    # ===== CSS =====
    st.markdown("""
    <style>
    .contact-card {
        background: #FFFFFF;
        padding: 1.3rem 1.4rem;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .profile-name {
        font-weight: 800;
        font-size: 1.15rem;
        color: #2D3436;
        margin-top: 0.6rem;
    }
    .profile-badge {
        display: inline-block;
        margin-top: 0.2rem;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(34,197,94,0.15);
        color: #166534;
        border: 1px solid rgba(34,197,94,0.35);
        font-weight: 600;
    }
    .contact-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        color: #2D3436;
    }
    .contact-sub {
        color: #636E72;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
    }
    .contact-item {
        margin: 0.45rem 0;
        font-size: 1rem;
    }
    .contact-item a {
        text-decoration: none;
        color: #0984E3;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Contact Me")
    st.caption("Questions, feedback, or collaboration? Reach out anytime.")

    left, right = st.columns([1.05, 1.25], gap="large")

    # ================= LEFT =================
    with left:

        # ===== FOTO + NAMA =====
        col_pic, col_info = st.columns([1, 2.2], gap="medium")

        with col_pic:
            st.image("foto.jpg", width=80)

        with col_info:
            st.markdown('<div class="profile-name">Mayra Nur Fatikha</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-badge">🟢 Open to collaborate</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ===== QUICK CONTACT CARD =====
        st.markdown(
            '<div class="contact-card">'
            '<div class="contact-title">📬 Quick Contact</div>'
            '<div class="contact-sub">Fastest ways to reach me</div>'

            '<div class="contact-item">📧 <b>Email:</b> '
            '<span id="email-text">mayrafatikha@gmail.com</span></div>'

            '<div class="contact-item">💼 <b>LinkedIn:</b> '
            '<a href="https://linkedin.com/in/mayranf" target="_blank">linkedin.com/in/mayranf</a></div>'

            '<div class="contact-item">🐙 <b>GitHub:</b> '
            '<a href="https://github.com/mayraatika" target="_blank">github.com/mayraatika</a></div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ===== COPY EMAIL BUTTON =====
        components.html(
            """
            <script>
            function copyEmail() {
                navigator.clipboard.writeText("mayrafatika@gmail.com");
                const btn = document.getElementById("copyBtn");
                btn.innerText = "Copied ✓";
                setTimeout(() => btn.innerText = "📋 Copy Email", 1800);
            }
            </script>

            <button id="copyBtn"
                onclick="copyEmail()"
                style="
                margin-top:8px;
                padding:8px 14px;
                border-radius:10px;
                border:1px solid #D1D5DB;
                background:#F9FAFB;
                cursor:pointer;
                font-weight:600;
                ">
                📋 Copy Email
            </button>
            """,
            height=55
        )

        st.info("✅ Available for project discussions, feedback, and data science collaboration.")

    # ================= RIGHT =================
    with right:
        st.markdown(
            '<div class="contact-card">'
            '<div class="contact-title">✍️ Send a Message</div>'
            '<div class="contact-sub">Leave a short note (demo form).</div>'
            '</div>',
            unsafe_allow_html=True
        )

        with st.form("contact_form", clear_on_submit=True):
            name = st.text_input("Name")
            email = st.text_input("Email")
            topic = st.selectbox(
                "Topic",
                ["Collaboration", "Feedback", "Question", "Other"]
            )
            message = st.text_area("Message", height=140)

            sent = st.form_submit_button("Send Message")

        if sent:
            if not name or not email or not message:
                st.warning("Please complete **Name**, **Email**, and **Message**.")
            else:
                st.success("Thanks! Your message has been recorded (demo). ✅")

    st.caption("Tip: LinkedIn is usually the fastest way to get a response.")
