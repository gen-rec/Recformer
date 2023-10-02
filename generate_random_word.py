from wonderwords import RandomWord


def main():
    random_word_generator = RandomWord()

    while True:
        random_word = random_word_generator.random_words(include_parts_of_speech=["noun", "verb"])[0]

        if " " in random_word or "-" in random_word:
            continue
        else:
            break

    print(random_word, end="")


if __name__ == "__main__":
    main()
