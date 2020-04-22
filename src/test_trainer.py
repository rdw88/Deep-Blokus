from unittest import TestCase, main
from trainer import BlokusTrainer


class BlokusTrainerTestCase(TestCase):
    def test_initialization(self):
        trainer = BlokusTrainer(
            output_file='models/test_model.h5',
            batch_size=100,
            num_steps=64,
            num_workers=4
        )

        self.assertEqual(trainer.output_file, 'models/test_model.h5')
        self.assertEqual(trainer.batch_size, 100)
        self.assertEqual(trainer.num_steps, 64)
        self.assertEqual(len(trainer.workers), 4)

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.game_queue)
        self.assertIsNotNone(trainer.training_complete_event)
        self.assertFalse(trainer.training_complete_event.is_set())

        with self.assertRaises(ValueError):
            BlokusTrainer(output_file='models/not_h5_file.txt')


    def test_get_batches(self):
        trainer = BlokusTrainer(
            output_file='models/test_model.h5',
            batch_size=3
        )

        trainer.enqueue_game([[1, 0], [0, 1]], [4, 13, 21, 1])
        trainer.enqueue_game([[1, 0], [0, 1]], [17, 21, 4, 13])
        trainer.enqueue_game([[1, 0], [0, 1]], [0, -10, 23, 1])

        batches = trainer.get_batches()
        inputs, outputs = next(batches)

        self.assertTupleEqual(inputs.shape, (3, 2))
        self.assertTupleEqual(outputs.shape, (3, 4))
        self.assertListEqual(inputs[0].tolist(), [1, 0])
        self.assertListEqual(inputs[1].tolist(), [0, 1])
        self.assertListEqual(inputs[2].tolist(), [1, 0])
        self.assertListEqual(outputs[0].tolist(), [0, 0, 0, 1])
        self.assertListEqual(outputs[1].tolist(), [0, 0, 0, 1])
        self.assertListEqual(outputs[2].tolist(), [0, 0, 1, 0])

        inputs, outputs = next(batches)

        self.assertTupleEqual(inputs.shape, (3, 2))
        self.assertTupleEqual(outputs.shape, (3, 4))
        self.assertListEqual(inputs[0].tolist(), [0, 1])
        self.assertListEqual(inputs[1].tolist(), [1, 0])
        self.assertListEqual(inputs[2].tolist(), [0, 1])
        self.assertListEqual(outputs[0].tolist(), [0, 0, 1, 0])
        self.assertListEqual(outputs[1].tolist(), [0, 1, 0, 0])
        self.assertListEqual(outputs[2].tolist(), [0, 1, 0, 0])


    def test_get_target(self):
        trainer = BlokusTrainer(output_file='models/test_model.h5')

        self.assertListEqual(trainer.get_target([4, 8, 3, 1]), [0, 0, 0, 1])
        self.assertListEqual(trainer.get_target([0, 12, 4, 5]), [1, 0, 0, 0])
        self.assertListEqual(trainer.get_target([0, -10, 12, 15]), [0, 1, 0, 0])


if __name__ == '__main__':
    main(warnings='ignore')