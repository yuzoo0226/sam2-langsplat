import os
import cv2
import base64
import openai
from icecream import ic


class OpenAIUtils():
    def __init__(self):
        # Get API key from environment variables
        ic("Connect for OpenAI server...")
        self.openai_key = os.environ.get("OPENAI_API")
        self.gpt_version = "gpt-4o"

        if self.openai_key is None:
            ic("OPENAI_API is not set.")
            # sys.exit()
        else:
            openai.api_key = self.openai_key
            if self._check_authenticate() is False:
                ic("OPENAI_API is set. But this key is invalid")
        self._reset_prompt()

    def _reset_prompt(self):
        self.messages = [
            {"role": "system", "content": "Please output various names that accurately expresses the characteristics of this object."},
            {"role": "system", "content": "If proper nouns such as product names are known, they should also be output as label names."},
            {"role": "system", "content": "The label name should include the shape, color, and other distinguishing characteristics."},
            {"role": "system", "content": "When labeling the object, give it a unique feature that does not conflict with other objects shown in previous images."},
            {"role": "system", "content": "The output should be in the form of [label1. label2. label3. label4. label5. ...]"},
        ]
        ic("prompt is reset")

    def _check_authenticate(self) -> bool:
        """Function to verify if the OpenAI API key is authenticated

        Returns:
            bool: True if authenticated, False otherwise
        """
        try:
            _ = openai.models.list()
        except openai.AuthenticationError:
            ic("OPENAI_API authentication failed")
            return False
        except Exception as e:
            ic(f"{e}")
            return False
        ic("OPENAI API authentication is successful")
        return True

    def _encode_image(self, image_path: str) -> str:
        """Function to encode the image in the format required by ChatGPT
        """
        _, image_extension = os.path.splitext(image_path)

        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            url = f"data:image/{image_extension};base64,{image_base64}"

        return url

    def _add_text_prompt(self, prompt: str, role: str = "user"):
        self.messages.append(
            {
                "role": role, "content":
                [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]
            }
        )

    def _add_image_prompt(self, image, image_path: str = "temp.png", prompt: str = None, role: str = "user"):
        cv2.imwrite(image_path, image)
        image_url = self._encode_image(image_path)
        if prompt is None:

            # 文字列と画像をプロンプトに追加
            self.messages.append(
                {
                    "role": role, "content":
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            )
        else:
            # 文字列と画像をプロンプトに追加
            self.messages.append(
                {
                    "role": role, "content":
                    [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            )

    def make_labels(self, image, image_path: str = "temp.png", prompt: str = None, save_answer: bool = True):
        self._add_image_prompt(image=image, image_path=image_path, prompt=prompt)

        # Prediction
        response = openai.chat.completions.create(
            model=self.gpt_version,
            messages=self.messages
        )

        answer = response.choices[0].message.content

        if save_answer:
            self.messages.append(
                {"role": "assistant", "content": answer}
            )
        else:
            _ = self.messages.pop()  # del image prompt

        answer_list = answer.strip("[]").split(". ")
        return answer_list
