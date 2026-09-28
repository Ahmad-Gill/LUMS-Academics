Model: `final_model.pt` | device: mps | max_new_tokens=100 | seeds=[42, 43, 44, 45, 46] | 3 prompts -> 15 samples per setting

| Setting | Mean new tokens | Ended with EOT | Distinct-2 (higher = more varied) | Repeated 4-grams (higher = more looping) |
|---|---:|---:|---:|---:|
| T=0.7, p=1.0 | 86.3 | 33% | 0.868 | 2.0% |
| T=1.0, p=1.0 | 97.7 | 13% | 0.918 | 0.5% |
| T=1.3, p=1.0 | 83.0 | 33% | 0.966 | 0.1% |
| T=1.0, p=0.8 | 85.9 | 20% | 0.891 | 1.4% |
| T=1.0, p=0.9 | 89.3 | 27% | 0.900 | 1.1% |

## Samples (seed 42, not cherry-picked)

### T=0.7, p=1.0

**Prompt 1 (Once upon a time):**

> Once upon a time, there was a little girl named Daisy. She was very upset. One day, she was playing with her toys when she noticed something in the corner. She saw it was a toy car. She picked it up and said to herself, "This is mine!"  She decided to show her mom. She said, "Mom, I found a toy car!" Her mom smiled and said, "That's nice, Daisy. But I think it's a special car. It's fake,

**Prompt 2 (The little girl):**

> The little girl was sad, but she understood that even if she didn't get the right thing, she could still make her mom happy.  When they got home, the little girl was happy again and she ran to her room to get some glue. She spent the rest of the day helping her mom and was very proud that she had helped her mom.<|endoftext|>

**Prompt 3 (One day, a small):**

> One day, a small boy named Tim went to the park. He saw a big dog. The dog was very friendly. The dog said, "I will go play with you." Tim wanted to play.  Tim took the dog with him to his home. He walked fast and had fun. Tim played near the dog. He had a fun day. When it was time to go home, Tim said, "Thank you for playing with me, dog. I had a good day at the park."  

### T=1.0, p=1.0

**Prompt 1 (Once upon a time):**

> Once upon a time, a little girl named Lily was on her way to school. She loved to play with her toys, but she was shy of the others to hold them.  When she came to the blackboard, she saw that her room had a boo-boo on it. She knew her mom was trying to fixes the violin, but she didn't know how. So, she asked her mom how to fix the violin.  Her mom explained that sometimes we hope to try again, but it

**Prompt 2 (The little girl):**

> The little girl was sad that she couldn't have another toy. But she then had an idea. She decided to melt the toy in his pot, while her mom baked her a special treat. She opened it and pulled out a mild result from the wood together! She was so happy that the broken toy could be fixed. Sara hugged it and wiped away the sad feeling that she could start her day with a bit of love to do it.<|endoftext|>

**Prompt 3 (One day, a small):**

> One day, a small boy saw the magical thing on his door. He wanted to say something. He opened the door and placed his hands on the key. Then the key started to open! He shrugged again and said, "I did it!" But then something he didn't know what it meant.   Suddenly, the door lights opened! The little boy quickly shut the door and the brilliant idea came to life. He was never heard from again! The boy continued he had found, but he never forgot the chance

### T=1.3, p=1.0

**Prompt 1 (Once upon a time):**

> Once upon a time, a little girl named Lily was on her way to school. She loved prunes and wanted to jump perfect low. But while waiting, the doorbell rang and Lily couldn't cry out.   Lily's mom recognized her worry as the police came because she had caught stuff suddenly passed. They realized that neither boys would permit her to jump in because necessary, while her mom protected her arms.   But Lily was determined to be just brave and she convinced herself to jump into the fit

**Prompt 2 (The little girl):**

> The little girl was sad that she couldn't have another mummy. But she insisted, because presents were heavy and big towels paid off. That's when you have her friend â€“ lots of pretend towels!  Despite as happy got away from home Abby and again, she could barely rub any arms she owned. Eventually so the tears that Sally had gotten up to leaving, but once it was empty than the mum had made for permission, it was her true complexed being too strong.  Unfortunately

**Prompt 3 (One day, a small):**

> One day, a small boy saw the magical wave in the sand. He was very clever.   The wave was scared, but the boy managed the wave as the wave scooped him up and drove town with him to safety. The wave safely replaced them and everyone cried in his way.   From the hill in the town kept big armgo the beautiful wave away to base cheer him on.<|endoftext|>

### T=1.0, p=0.8

**Prompt 1 (Once upon a time):**

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her favorite toy was a teddy bear. One day, Lily's mommy asked her to help prepare a special dish for the teddy bear.   Lily was very happy to help and started putting on a big apron. She turned on the stove and put a bow on the teddy bear. She felt very grown-up and didn't like it.   After a while, Lily's mommy made her a yummy soup

**Prompt 2 (The little girl):**

> The little girl was sad, but she understood that even if she didn't want to play with the yellow ball, it was still fun to have fun and enjoy the sunshine.<|endoftext|>

**Prompt 3 (One day, a small):**

> One day, a small boy came to the park. He wanted to play with his ball, but he was shy. He looked around and saw a cat. The cat was sitting on a branch. The boy was scared of the cat.  "Hi, cat!" he said. "Can I play with you?"  The cat smiled and said, "Yes, you can. But be careful, the cat is very gentle. You don't want to scare me."  The boy touched the cat gently

### T=1.0, p=0.9

**Prompt 1 (Once upon a time):**

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her favorite toy was a teddy bear. One day, Lily's mommy asked her to help prepare a special dish for the teddy bear.   Lily was very happy to help and started putting on a big apron. She turned on the stove and put a bow on the teddy bear's head. The teddy bear smelled so good!   After the table was finished, Lily took it outside to play with her friends

**Prompt 2 (The little girl):**

> The little girl was sad that she couldn't have another toy. But she didn't have a spoiled toy. She had a bad ending.<|endoftext|>

**Prompt 3 (One day, a small):**

> One day, a small boy saw the marble and ran over to pick it up. He held it in his hand and smiled at it. He looked very happy to find it and he wanted to keep it for himself.  The boy and the boy became very good friends. They played together all day. They laughed, ran, and they had lots of fun together.  The end.<|endoftext|>
